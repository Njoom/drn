import argparse
import shutil
import time

import numpy as np
import os
from os.path import exists, split, join, splitext

import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import zipfile

import drn as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test', 'map', 'locate'])
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='drn18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: drn18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--check-freq', default=10, type=int,
                        metavar='N', help='checkpoint frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--lr-adjust', dest='lr_adjust',
                        choices=['linear', 'step'], default='step')
    parser.add_argument('--crop-size', dest='crop_size', type=int, default=224)
    parser.add_argument('--scale-size', dest='scale_size', type=int, default=256)
    parser.add_argument('--step-ratio', dest='step_ratio', type=float, default=0.1)
    args = parser.parse_args()
    return args


def main():
    print(' '.join(sys.argv))
    args = parse_args()
    print(args)
    if args.cmd == 'train':
        run_training(args)
    elif args.cmd == 'test':
        test_model(args)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================
# ZIP / FOLDER helper
# ==========================
def prepare_data_root(data_path):
    """
    If data_path is a directory: return it as-is.
    If data_path is a .zip file (e.g. /content/drive/MyDrive/Wang_CVPR2020.zip):
      - Extract it into /content
      - Expect a folder /content/Wang_CVPR2020/ containing:
            training/
            validation/
            testing/
      - Return that folder as data_root.
    This keeps all heavy I/O on Colab local disk, not on Drive.
    """
    # Case 1: already a directory
    if os.path.isdir(data_path):
        print(f"[INFO] Using existing directory as data root: {data_path}")
        return data_path

    # Case 2: ZIP file (your Wang_CVPR2020.zip case)
    if os.path.isfile(data_path) and data_path.lower().endswith(".zip"):
        zip_abs = os.path.abspath(data_path)
        zip_base = os.path.splitext(os.path.basename(zip_abs))[0]  # e.g. "Wang_CVPR2020"

        # We always extract into /content
        extract_root = "/content"
        target_dir = os.path.join(extract_root, zip_base)  # /content/Wang_CVPR2020

        # If target_dir doesn't exist or is empty -> extract
        if not os.path.isdir(target_dir) or len(os.listdir(target_dir)) == 0:
            print(f"[INFO] Extracting ZIP dataset from: {zip_abs}")
            with zipfile.ZipFile(zip_abs, "r") as zf:
                zf.extractall(extract_root)
            print(f"[INFO] Extraction done under: {extract_root}")
        else:
            print(f"[INFO] Using already extracted data under: {target_dir}")

        # Now we expect: /content/Wang_CVPR2020/training, /validation, /testing
        if os.path.isdir(os.path.join(target_dir, "training")):
            print(f"[INFO] Using data root: {target_dir}")
            return target_dir
        else:
            # Fallback (rare): training not inside /content/Wang_CVPR2020
            print(f"[WARN] 'training' folder not found inside {target_dir}. "
                  f"Using {extract_root} as data root.")
            return extract_root

    # Not a folder and not a ZIP
    raise ValueError(f"Data path '{data_path}' is neither a directory nor a .zip file.")


class ButterworthHighpass2D(nn.Module):
    """
    Butterworth high-pass filter in the frequency domain, applied per image.

    Input:  x of shape (B, C, H, W)
    Output: same shape (B, C, H, W)
    """

    def __init__(self, D0=80.0, n=2):
        super(ButterworthHighpass2D, self).__init__()
        self.D0 = float(D0)
        self.n = int(n)
        # cache filter (not saved in state_dict)
        self.register_buffer("H_filter", None, persistent=False)

    def _make_filter(self, H, W, device, dtype):
        # Reuse cached filter if size matches
        if self.H_filter is not None and self.H_filter.shape[-2:] == (H, W):
            return self.H_filter

        u = torch.arange(H, device=device, dtype=dtype) - H / 2.0
        v = torch.arange(W, device=device, dtype=dtype) - W / 2.0
        V, U = torch.meshgrid(v, u)  # (W,H) each
        U = U.t()
        V = V.t()

        D = torch.sqrt(U ** 2 + V ** 2)  # (H, W)
        eps = 1e-8
        Hf = 1.0 / (1.0 + (self.D0 / (D + eps)) ** (2 * self.n))  # Butterworth HPF
        Hf = torch.fft.ifftshift(Hf)  # match unshifted fft2

        Hf = Hf.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        self.H_filter = Hf
        return Hf

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        assert x.dim() == 4, "Input must be (B, C, H, W)"
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        Hf = self._make_filter(H, W, device, dtype)  # (1,1,H,W)
        # Broadcast to (B, C, H, W) via broadcasting rules
        Xf = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        Yf = Xf * Hf  # broadcast over batch & channels
        y = torch.fft.ifft2(Yf, dim=(-2, -1), norm="ortho")
        y = torch.real(y)
        return y


class DRNWithButterworth(nn.Module):
    """
    Wrapper around a DRN model that concatenates:
      - original RGB image
      - Butterworth high-pass filtered image

    Pipeline:
      x_rgb -> HPF(x_rgb) -> cat([x_rgb, HPF(x_rgb)], dim=1) -> Conv(6->3)+ReLU -> DRN
    """

    def __init__(self, base_model, D0=60.0, n=2):
        super(DRNWithButterworth, self).__init__()
        self.base_model = base_model
        self.hpf = ButterworthHighpass2D(D0=D0, n=n)

        # Fuse the 6-channel tensor (RGB + HPF RGB) back to 3 channels
        self.fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        # Kaiming init for fuse conv
        nn.init.kaiming_normal_(self.fuse[0].weight, mode='fan_out', nonlinearity='relu')
        if self.fuse[0].bias is not None:
            nn.init.constant_(self.fuse[0].bias, 0.0)

    def forward(self, x):
        # x is normalized RGB: shape (B, 3, H, W)
        x_rgb = x
        x_hpf = self.hpf(x_rgb)                 # (B, 3, H, W)

        x_cat = torch.cat([x_rgb, x_hpf], dim=1)  # (B, 6, H, W)
        x_fused = self.fuse(x_cat)               # (B, 3, H, W)

        # Feed into original DRN (unchanged)
        out = self.base_model(x_fused)
        return out



def run_training(args):
    # ----- create base DRN model ----- 
    base_model = models.__dict__[args.arch](args.pretrained)

    # ----- adjust classifier to 2 classes ----- 
    NUM_CLASSES = 2
    in_channels = base_model.out_dim          # DRN defines this as last feature dim
    base_model.fc = nn.Conv2d(
        in_channels,
        NUM_CLASSES,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True
    )
    nn.init.kaiming_normal_(base_model.fc.weight, mode='fan_out', nonlinearity='relu')
    if base_model.fc.bias is not None:
        nn.init.constant_(base_model.fc.bias, 0.)

    # ----- wrap DRN with Butterworth + concat block ----- 
    # You can tune D0 and n if you like
    model = DRNWithButterworth(base_model, D0=30.0, n=2)

    # ----- move to GPU & DataParallel ----- 
    model = torch.nn.DataParallel(model).cuda()


    best_prec1 = 0.0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    # ------------------------------
    # Data root: handle ZIP or folder
    # ------------------------------
    data_root = prepare_data_root(args.data)

    # Data loading code
    traindir = os.path.join(data_root, 'training')
    valdir = os.path.join(data_root, 'validation')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])



    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # to check:
    print("Train classes:", train_loader.dataset.classes)
    print("Train class_to_idx:", train_loader.dataset.class_to_idx)
    print("Val classes:", val_loader.dataset.classes)
    print("Val class_to_idx:", val_loader.dataset.class_to_idx)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(args, val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        checkpoint_path = 'checkpoint_latest.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        if (epoch + 1) % args.check_freq == 0:
            history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)


def test_model(args):
    # base DRN model
    base_model = models.__dict__[args.arch](args.pretrained)

    # adjust classifier to 2 classes
    NUM_CLASSES = 2
    in_channels = base_model.out_dim
    base_model.fc = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1, stride=1, padding=0, bias=True)
    nn.init.kaiming_normal_(base_model.fc.weight, mode='fan_out', nonlinearity='relu')
    if base_model.fc.bias is not None:
        nn.init.constant_(base_model.fc.bias, 0.)

    # wrap DRN with Butterworth + concat block
    model = DRNWithButterworth(base_model, D0=30.0, n=2)

    # move to GPU & wrap
    model = torch.nn.DataParallel(model).cuda()


    # Optionally load a checkpoint from --resume (for compatibility)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

   
    data_root = prepare_data_root(args.data)

    # Data loading code
    # NOTE: your testing path: data_root/testing/crn
    testdir = os.path.join(data_root, 'testing', 'crn')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    
    
    

    t = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, t),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print("Test classes:", test_loader.dataset.classes)
    print("Test class_to_idx:", test_loader.dataset.class_to_idx)

    criterion = nn.CrossEntropyLoss().cuda()

    # ---- evaluate LAST and BEST checkpoints with full metrics ----
    num_params = count_parameters(model)
    results = {}

    def load_and_eval(ckpt_path, tag):
        if not os.path.isfile(ckpt_path):
            print(f"[WARN] Checkpoint not found: {ckpt_path}")
            return

        print(f"\n=== Evaluating {tag} checkpoint: {ckpt_path} ===")
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])

        metrics = eval_loader_with_metrics(
            test_loader, model, criterion, split_name="TEST"
        )
        metrics["params"] = num_params
        results[tag] = metrics

    # Paths used during training
    last_ckpt = 'checkpoint_latest.pth.tar'
    best_ckpt = 'model_best.pth.tar'

    # 1) LAST epoch checkpoint
    load_and_eval(last_ckpt, tag="LAST")

    # 2) BEST validation checkpoint
    load_and_eval(best_ckpt, tag="BEST")

    # Print final table
    print_results_table(results)


def train(args, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input  = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        input_var = input
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1,  = accuracy(output.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


def validate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        input  = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def eval_loader_with_metrics(loader, model, criterion, split_name="TEST"):
    """
    Evaluate model on a loader and compute:
    accuracy, sensitivity, specificity, loss, and computation time.
    Assumes binary classification with labels {0,1}, where:
    - Positive class = 1 (fake)
    - Negative class = 0 (real)
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Confusion matrix components
    TP = TN = FP = FN = 0

    start_time = time.time()

    with torch.no_grad():
        for input, target in loader:
            input  = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(input)
            loss = criterion(output, target)

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size

            _, preds = torch.max(output, 1)
            total_correct += torch.sum(preds == target).item()
            total_samples += batch_size

            # Update confusion matrix (binary)
            for t, p in zip(target.cpu().numpy(), preds.cpu().numpy()):
                if t == 1 and p == 1:
                    TP += 1
                elif t == 0 and p == 0:
                    TN += 1
                elif t == 0 and p == 1:
                    FP += 1
                elif t == 1 and p == 0:
                    FN += 1

    elapsed = time.time() - start_time

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    print(f"{split_name}  Loss: {avg_loss:.4f}  Acc: {accuracy:.4f}")
    print(f"{split_name}  Sensitivity (TPR): {sensitivity:.4f}")
    print(f"{split_name}  Specificity (TNR): {specificity:.4f}")
    print(f"{split_name}  Time: {elapsed:.2f} sec")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "time": elapsed,
    }


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by step_ratio every 30 epochs"""
    lr = args.lr * (args.step_ratio ** (epoch // 30))
    print('Epoch [{}] Learning rate: {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def print_results_table(results):
    """
    Print a copy-friendly table of metrics for different checkpoints.
    results: dict[tag] = {
        'accuracy', 'sensitivity', 'specificity', 'loss', 'time', 'params'
    }
    """
    print("\n================ FINAL TEST RESULTS =================")
    print(f"{'Model':<8} | {'Acc':<8} | {'Sens':<8} | {'Spec':<8} | "
          f"{'Loss':<8} | {'Time(s)':<8} | {'Params(M)':<10}")
    print("-" * 90)

    for name, r in results.items():
        print(
            f"{name:<8} | "
            f"{r['accuracy']:.4f} | "
            f"{r['sensitivity']:.4f} | "
            f"{r['specificity']:.4f} | "
            f"{r['loss']:.4f} | "
            f"{r['time']:.2f} | "
            f"{r['params'] / 1e6:.2f}"
        )


if __name__ == '__main__':
    main()
