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

import torch.nn.functional as F

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

class HFRFWrapper(nn.Module):
    """
    Wraps a base DRN model and applies ONLY the hfreqWH block
    (high-frequency filtering over spatial dimensions) on the INPUT IMAGE
    before feeding it to DRN.

    x -> hfreqWH -> base_model(x)
    """

    def __init__(self, base_model, scale_wh=4):
        super(HFRFWrapper, self).__init__()
        self.base_model = base_model
        self.scale_wh = scale_wh

    # ---------- HFRI: high-frequency over spatial dims (W,H) ----------
    def hfreqWH(self, x, scale: int):
        """
        x: (B, C, H, W)
        High-pass filter in spatial frequency domain.
        """
        assert scale > 2, "scale must be > 2"

        # 2D FFT over H, W
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])

        b, c, h, w = x.shape

        # Zero out LOW-frequency center block
        x[:, :,
          h // 2 - h // scale : h // 2 + h // scale,
          w // 2 - w // scale : w // 2 + w // scale] = 0.0

        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")

        x = torch.real(x)
        x = F.relu(x, inplace=True)
        return x

    def forward(self, x):
        # Apply high-frequency spatial filter once in the forward pass
        x = self.hfreqWH(x, self.scale_wh)

        # Then feed to the original DRN model
        x = self.base_model(x)
        return x







    

def run_training(args):
    # ----- create base model on CPU ----- 
    base_model = models.__dict__[args.arch](args.pretrained)

    # ----- adjust classifier to 2 classes BEFORE wrapping ----- 
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

    # ----- Wrap with hfreqWH-only block -----
    # This applies HFRI (spatial high-frequency) to the input image
    model = HFRFWrapper(base_model, scale_wh=4)

    # ----- NOW move to GPU & DataParallel -----
    model = torch.nn.DataParallel(model).cuda()
    ...

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

    # Data loading code
    traindir = os.path.join(args.data, 'training')
    valdir = os.path.join(args.data, 'validation')
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
    #to check:
    print("Train classes:", train_loader.dataset.classes)
    print("Train class_to_idx:", train_loader.dataset.class_to_idx)
    print("Val classes:", val_loader.dataset.classes)
    print("Val class_to_idx:", val_loader.dataset.class_to_idx)


    # define loss function (criterion) and pptimizer
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
    
    # base model on CPU
    base_model = models.__dict__[args.arch](args.pretrained)

    # adjust classifier to 2 classes
    NUM_CLASSES = 2
    in_channels = base_model.out_dim
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

    # wrap with hfreqWH-only block
    model = HFRFWrapper(base_model, scale_wh=4)

    # move to GPU & wrap
    model = torch.nn.DataParallel(model).cuda()
    ...



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

    # Data loading code
    testdir = os.path.join(args.data, 'testing', 'crn')
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

    # ---- NEW: evaluate LAST and BEST checkpoints with full metrics ----
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
    #top5 = AverageMeter()

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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
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
