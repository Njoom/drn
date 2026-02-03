Skip to content
Chat history


You said:
regenerate the last answer
ChatGPT said:
أكيد 🌟
هذه إعادة توليد للإجابة السابقة بشكل أوضح ومُنظم، وبأسلوب مناسب لعملك البحثي في الـ Deep Learning / Deepfake Detection:

🔍 رأيي في أداء Intel Arc™ 140T GPU لأعمال الذكاء الاصطناعي

https://img.geekom.it/geekomit/2025/08/intel-arc-140t-ray-tracing.webp

https://imageio.forbes.com/specials-images/imageserve//62965cf3efbd9026d0d94143/0x0.jpg?fit=bounds&format=jpg&height=600&width=1200

https://gamersnexus.net/u/styles/large_responsive_no_watermark_/public/inline-images/GN%20GPU%20Benchmark%20Dying%20Light%202%20Stay%20Human%20%281440p_High-Custom%29%20GamersNexus-4x_foolhardy_Remacri_1.png.webp
4
🧠 نظرة عامة
Intel Arc™ 140T هو كرت شاشة مخصص للأجهزة المحمولة (Laptop GPU)، وموجه أكثر للأعمال العامة والـ graphics، مع دعم محدود نسبيًا لتطبيقات Deep Learning المتقدمة.

⚙️ المواصفات (تقريبية)
🧮 VRAM: من 6GB إلى 8GB

⚡ Architecture: Intel Xe-HPG

🧩 XMX Cores: مدعومة (مفيدة لـ Intel OpenVINO)

🔌 CUDA: ❌ غير مدعوم

🔥 ROCm: ❌ غير مدعوم

🧠 TensorRT: ❌ غير مدعوم

✅ متى يكون مناسب لك؟
مناسب جزئيًا فقط إذا كان استخدامك:

✔️ تجارب أولية (Prototyping)

✔️ تدريب Models صغيرة

✔️ Inference خفيف

✔️ استخدام Intel OpenVINO

✔️ العمل على:

CNNs خفيفة

Fine-tuning بسيط

Preprocessing / Feature extraction

❌ متى لا أنصح به؟
❌ غير مناسب لحالتك البحثية الحالية إذا كنتِ:

تعملين على Deepfake Detection متقدم

تستخدمين:

ResNet / DRN / ViT

FFT / FreqNet

Multi-stage pipelines

تحتاجين:

Batch size كبير

Training طويل

GPU RAM عالية (أنتِ ذكرتِ سابقًا أنك تحتاجين حتى ~84GB)

⚠️ مشاكل عملية ستواجهينها
أغلب كود PyTorch مكتوب على CUDA

كثير من Repos على GitHub:

❌ لا تعمل مباشرة على Intel Arc

❌ تحتاج Workarounds

Debugging أعلى

Performance غير مستقر في Training مقارنة بـ NVIDIA

🟢 المقارنة السريعة (مهمّة لك)
GPU	مناسب لـ Deep Learning؟	ملاحظات
Intel Arc 140T	❌ لا	جيد للاستخدام العام فقط
NVIDIA RTX 3060 (12GB)	✅ نعم	الحد الأدنى الموصى به
NVIDIA RTX 3080 / 3090	✅✅	ممتاز للأبحاث
NVIDIA RTX 4090	⭐⭐⭐	الأفضل حاليًا
Cloud (A100 / H100)	⭐⭐⭐⭐	مناسب لأبحاث الدكتوراه
🎓 نصيحتي لك (كبروفايل باحثة دكتوراه)
بما أن:

شغلك بحثي متقدم

تدمجين FFT + CNN + DRN

عندك تجارب ثقيلة وطويلة

وتحتاجين نتائج موثوقة للنشر

👉 لا تعتمدي على Intel Arc 140T كجهازك الأساسي
👉 استخدميه فقط كحل مؤقت أو مساعد

✅ الحل الأمثل لك
💻 Laptop + NVIDIA RTX حقيقي

أو ☁️ Google Colab Pro / Kaggle / Cloud GPU

أو 🖥️ Workstation خارجي

لو حابة:

أرشح لك Laptop محدد

أو Workstation بمواصفات دقيقة

أو أفضل خيار Cloud حسب ميزانيتك

قولي لي، وأنا أرتّبها لك مباشرة 💙

You said:
i want to run my code on the colab cpu. the code is to preprocessed images. i will preprocessed around 15000 images. the images is on zip folder on drive. this zip file had 100 folders each had 300-400 images. so i want you to make modifiation on my code so/;1- the input folder replaced with zip file.2-extract the zip file annd run the code folder by folder with the help of cpu multi processor. make the saving results at once for each folder not image by image. 3-create a csv file for each folder has all the info of the 68 detection points for each image in the folder. thus is the code (# this is an updated version of the code and its the perfect one:
# =============================
# Simple Processing (save only final images)
# Steps kept:
# 1) Face detection (SFD via face-alignment)
# 2) Choose largest face
# 3) Offset step (margin bounding box with skew)
# 4) Crop extended bbox
# 5) Isotropic resize by fixing the LARGEST side to 224
# 6) Center on 224×224 canvas (keeps output size fixed)
# =============================

!pip -q install face-alignment==1.3.5 opencv-python==4.10.0.84 imageio==2.35.1

import os
import glob
import cv2
import torch
import math
import numpy as np
from tqdm import tqdm
import face_alignment

# =============================
# Config (UPDATED)
# =============================
INPUT_DIR  = "/content/drive/MyDrive/Datasets/Experiments_Preprocessing/00"
OUTPUT_DIR = "/content/drive/MyDrive/Datasets/Experiments_Preprocessing/SimpleProcessing/fake_inpainting_00"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Padding controls
MARGIN_X = 1.3
MARGIN_Y = 1.6
SKEW_UP  = 2/3  # 2/3 padding above, 1/3 below

TARGET = 224

# Face-alignment init
device = 'cuda' if torch.cuda.is_available() else 'cpu'
FA = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device=device,
    flip_input=False
)

# =============================
# Helpers
# =============================
def clamp_box(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0, min(int(round(xmin)), w - 1))
    ymin = max(0, min(int(round(ymin)), h - 1))
    xmax = max(0, min(int(round(xmax)), w - 1))
    ymax = max(0, min(int(round(ymax)), h - 1))
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)
    return xmin, ymin, xmax, ymax

def largest_face(landmarks_list):
    if len(landmarks_list) == 1:
        return landmarks_list[0]
    best = landmarks_list[0]
    best_area = -1
    for lm in landmarks_list:
        hull = cv2.convexHull(lm.astype(np.float32))
        area = cv2.contourArea(hull)
        if area > best_area:
            best_area = area
            best = lm
    return best

import math
import cv2
import numpy as np

def crop_and_iso_resize_to_224_no_padding(bgr, x1, y1, x2, y2, target=224, interpolation=cv2.INTER_AREA):
    """
    1) Crop bbox.
    2) Isotropic resize so SHORTER side becomes 224 (so both sides >= 224).
    3) Center-crop the longer side to 224 using floor/ceil split (no padding).
    Returns exactly (224,224,3) or None.
    """
    crop = bgr[y1:y2+1, x1:x2+1]
    if crop.size == 0:
        return None

    ch, cw = crop.shape[:2]
    if ch <= 1 or cw <= 1:
        return None

    # IMPORTANT: scale by SHORTER side, not max side
    scale = float(target) / float(min(ch, cw))
    new_w = int(round(cw * scale))
    new_h = int(round(ch * scale))
    new_w = max(target, new_w)  # ensure >= target (guard against rounding)
    new_h = max(target, new_h)

    resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
    H, W = resized.shape[:2]

    # If height is longer than target -> crop rows
    if H > target:
        delta = H - target
        r1 = int(math.floor(delta / 2.0))
        r2 = int(math.ceil(delta / 2.0))
        resized = resized[r1:H - r2, :]
        H, W = resized.shape[:2]  # update

    # If width is longer than target -> crop cols
    if W > target:
        delta = W - target
        c1 = int(math.floor(delta / 2.0))
        c2 = int(math.ceil(delta / 2.0))
        resized = resized[:, c1:W - c2]
        H, W = resized.shape[:2]  # update

    # Safety: enforce exact 224×224 (rare rounding issues)
    if resized.shape[0] != target or resized.shape[1] != target:
        # tiny fallback (no padding): force exact size
        resized = cv2.resize(resized, (target, target), interpolation=interpolation)

    return resized




# =============================
# Collect images
# =============================
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
image_paths = sorted(
    p for p in glob.glob(os.path.join(INPUT_DIR, "*"))
    if os.path.splitext(p)[1].lower() in valid_ext
)

if not image_paths:
    print(f"No images found in {INPUT_DIR}")
else:
    print(f"Found {len(image_paths)} images in {INPUT_DIR}")

# =============================
# Processing loop (SAVE ONLY final images)
# =============================
ok = fail = 0

for img_path in tqdm(image_paths):
    fname = os.path.basename(img_path)
    base, _ = os.path.splitext(fname)

    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"[WARN] Could not read {fname}")
        fail += 1
        continue

    h, w = bgr.shape[:2]

    # 1) Face detection
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lms_list = FA.get_landmarks_from_image(rgb)
    if not lms_list:
        print(f"[WARN] No face detected in {fname}")
        fail += 1
        continue

    # 2) Choose largest face
    landmarks = largest_face(lms_list)

    # 3) Tight bbox from landmarks
    x_min, y_min = float(np.min(landmarks[:, 0])), float(np.min(landmarks[:, 1]))
    x_max, y_max = float(np.max(landmarks[:, 0])), float(np.max(landmarks[:, 1]))

    # 4) Offset/margin bbox
    w_box = x_max - x_min
    h_box = y_max - y_min

    left_extra   = 0.5 * MARGIN_X * w_box
    right_extra  = 0.5 * MARGIN_X * w_box
    top_extra    = SKEW_UP * MARGIN_Y * h_box
    bottom_extra = (1 - SKEW_UP) * MARGIN_Y * h_box

    xp_min = x_min - left_extra
    xp_max = x_max + right_extra
    yp_min = y_min - top_extra
    yp_max = y_max + bottom_extra

    xp_min, yp_min, xp_max, yp_max = clamp_box(xp_min, yp_min, xp_max, yp_max, w, h)

    # 5) Crop extended bbox + isotropic resize (largest side -> 224) + center to 224×224
    out_img = crop_and_iso_resize_to_224_no_padding(
        bgr, xp_min, yp_min, xp_max, yp_max,
        target=TARGET, interpolation=cv2.INTER_AREA
    )
    if out_img is None:
        print(f"[WARN] Empty/invalid crop after resize for {fname}")
        fail += 1
        continue

    # 6) Save final image
    out_path = os.path.join(OUTPUT_DIR, f"{base}.jpg")
    if cv2.imwrite(out_path, out_img):
        ok += 1
    else:
        print(f"[ERR] Failed saving {out_path}")
        fail += 1

print(f"✅ Done. Saved to: {OUTPUT_DIR} | OK={ok}, Fail={fail}") ).and this is the detecor (!pip install face-alignment)
ChatGPT said:

Thought for 22s
Nice, this is a perfect use case for CPU + multiprocessing on Colab 👍
I’ll give you one complete code cell that does all of this:

Reads a ZIP file from Drive (instead of a folder)

Extracts it to /content/extracted_dataset

Processes folder by folder in parallel using CPU multiprocessing

For each folder:

Saves the processed images into a matching folder under OUTPUT_ROOT

Creates one CSV file with the 68 landmark points for each successfully processed image

You can then just adjust the paths (ZIP path + output root) and run.

✅ Full modified code (ready for Colab, CPU, ZIP, multiprocessing, CSV per folder)
# ==========================================
# Simple Processing + ZIP + Multiprocessing + CSV per folder (CPU only)
# ==========================================

!pip -q install face-alignment==1.3.5 opencv-python==4.10.0.84 imageio==2.35.1

import os
import glob
import cv2
import torch
import math
import numpy as np
from tqdm import tqdm
import face_alignment
import zipfile
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# CONFIG (EDIT THESE PATHS)
# ==========================================
# 1) Path to your ZIP file on Drive
ZIP_PATH = "/content/drive/MyDrive/Datasets/Experiments_Preprocessing/your_dataset.zip"

# 2) Where to extract the ZIP inside Colab
EXTRACT_ROOT = "/content/extracted_dataset"

# 3) Where to save processed images + CSVs on Drive
OUTPUT_ROOT = "/content/drive/MyDrive/Datasets/Experiments_Preprocessing/SimpleProcessing_from_zip"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Padding controls (same as your original code)
MARGIN_X = 1.3
MARGIN_Y = 1.6
SKEW_UP  = 2/3  # 2/3 padding above, 1/3 below

TARGET = 224

# Force CPU (as you requested)
device = 'cpu'
torch.set_num_threads(os.cpu_count())


# ==========================================
# Helper functions
# ==========================================
def clamp_box(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0, min(int(round(xmin)), w - 1))
    ymin = max(0, min(int(round(ymin)), h - 1))
    xmax = max(0, min(int(round(xmax)), w - 1))
    ymax = max(0, min(int(round(ymax)), h - 1))
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)
    return xmin, ymin, xmax, ymax


def largest_face(landmarks_list):
    if len(landmarks_list) == 1:
        return landmarks_list[0]
    best = landmarks_list[0]
    best_area = -1
    for lm in landmarks_list:
        hull = cv2.convexHull(lm.astype(np.float32))
        area = cv2.contourArea(hull)
        if area > best_area:
            best_area = area
            best = lm
    return best


def crop_and_iso_resize_to_224_no_padding(bgr, x1, y1, x2, y2, target=224, interpolation=cv2.INTER_AREA):
    """
    1) Crop bbox.
    2) Isotropic resize so SHORTER side becomes 224 (so both sides >= 224).
    3) Center-crop the longer side to 224 using floor/ceil split (no padding).
    Returns exactly (224,224,3) or None.
    """
    crop = bgr[y1:y2+1, x1:x2+1]
    if crop.size == 0:
        return None

    ch, cw = crop.shape[:2]
    if ch <= 1 or cw <= 1:
        return None

    # Scale by SHORTER side
    scale = float(target) / float(min(ch, cw))
    new_w = int(round(cw * scale))
    new_h = int(round(ch * scale))
    new_w = max(target, new_w)
    new_h = max(target, new_h)

    resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
    H, W = resized.shape[:2]

    # If height is longer than target -> crop rows
    if H > target:
        delta = H - target
        r1 = int(math.floor(delta / 2.0))
        r2 = int(math.ceil(delta / 2.0))
        resized = resized[r1:H - r2, :]
        H, W = resized.shape[:2]

    # If width is longer than target -> crop cols
    if W > target:
        delta = W - target
        c1 = int(math.floor(delta / 2.0))
        c2 = int(math.ceil(delta / 2.0))
        resized = resized[:, c1:W - c2]
        H, W = resized.shape[:2]

    # Safety: enforce exact 224×224 (rare rounding issues)
    if resized.shape[0] != target or resized.shape[1] != target:
        resized = cv2.resize(resized, (target, target), interpolation=interpolation)

    return resized


def init_face_alignment():
    """Create a face-alignment detector on CPU."""
    FA = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device='cpu',
        flip_input=False
    )
    return FA


# ==========================================
# Folder processing function (runs in each process)
# ==========================================
def process_single_folder(folder_path):
    """
    Process all images in one folder:
      - detect face
      - crop + resize
      - save final image
      - collect 68 landmarks per image
      - write ONE CSV file for this folder

    Returns: (folder_path, ok_count, fail_count, num_rows_in_csv)
    """
    FA = init_face_alignment()

    # Compute relative folder name to recreate structure under OUTPUT_ROOT
    rel_folder = os.path.relpath(folder_path, EXTRACT_ROOT)
    out_folder = os.path.join(OUTPUT_ROOT, rel_folder)
    os.makedirs(out_folder, exist_ok=True)

    # CSV path for this folder
    csv_path = os.path.join(out_folder, "landmarks_68pts.csv")

    # Collect images in this folder only (no recursion)
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    image_paths = sorted(
        p for p in glob.glob(os.path.join(folder_path, "*"))
        if os.path.splitext(p)[1].lower() in valid_ext
    )

    ok = 0
    fail = 0
    rows = []

    if not image_paths:
        # Nothing to do for this folder
        return (folder_path, ok, fail, 0)

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        base, _ = os.path.splitext(fname)

        bgr = cv2.imread(img_path)
        if bgr is None:
            fail += 1
            continue

        h, w = bgr.shape[:2]

        # 1) Face detection
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lms_list = FA.get_landmarks_from_image(rgb)
        if not lms_list:
            fail += 1
            continue

        # 2) Choose largest face
        landmarks = largest_face(lms_list)  # (68, 2)

        # 3) Tight bbox from landmarks
        x_min, y_min = float(np.min(landmarks[:, 0])), float(np.min(landmarks[:, 1]))
        x_max, y_max = float(np.max(landmarks[:, 0])), float(np.max(landmarks[:, 1]))

        # 4) Offset/margin bbox
        w_box = x_max - x_min
        h_box = y_max - y_min

        left_extra   = 0.5 * MARGIN_X * w_box
        right_extra  = 0.5 * MARGIN_X * w_box
        top_extra    = SKEW_UP * MARGIN_Y * h_box
        bottom_extra = (1 - SKEW_UP) * MARGIN_Y * h_box

        xp_min = x_min - left_extra
        xp_max = x_max + right_extra
        yp_min = y_min - top_extra
        yp_max = y_max + bottom_extra

        xp_min, yp_min, xp_max, yp_max = clamp_box(xp_min, yp_min, xp_max, yp_max, w, h)

        # 5) Crop extended bbox + isotropic resize (no padding)
        out_img = crop_and_iso_resize_to_224_no_padding(
            bgr, xp_min, yp_min, xp_max, yp_max,
            target=TARGET, interpolation=cv2.INTER_AREA
        )
        if out_img is None:
            fail += 1
            continue

        # 6) Save final image (inside this folder's output)
        out_path = os.path.join(out_folder, f"{base}.jpg")
        saved = cv2.imwrite(out_path, out_img)
        if not saved:
            fail += 1
            continue

        # 7) Collect 68-point landmarks into CSV row
        #    Row format: image_name, x_0 ... x_67, y_0 ... y_67
        xs = landmarks[:, 0].tolist()
        ys = landmarks[:, 1].tolist()
        row = [fname] + xs + ys
        rows.append(row)
        ok += 1

    # Write CSV ONCE per folder
    if rows:
        header = (["image_name"] +
                  [f"x_{i}" for i in range(68)] +
                  [f"y_{i}" for i in range(68)])
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    return (folder_path, ok, fail, len(rows))


# ==========================================
# 1) Extract ZIP (only once)
# ==========================================
if not os.path.isdir(EXTRACT_ROOT) or len(os.listdir(EXTRACT_ROOT)) == 0:
    print(f"Extracting ZIP from: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_ROOT)
    print("✅ Extraction done.")
else:
    print(f"EXTRACT_ROOT already has files, skipping extraction: {EXTRACT_ROOT}")


# ==========================================
# 2) Collect first-level folders to process
#    (Assuming your ZIP has ~100 folders at top level)
# ==========================================
all_folders = sorted(
    p for p in glob.glob(os.path.join(EXTRACT_ROOT, "*"))
    if os.path.isdir(p)
)

print(f"Found {len(all_folders)} folders to process under {EXTRACT_ROOT}.")

# ==========================================
# 3) Run multiprocessing over folders (CPU)
# ==========================================
if __name__ == "__main__":
    total_ok = 0
    total_fail = 0
    total_rows = 0

    # You can reduce max_workers (e.g., 2 or 4) if memory is tight
    max_workers = min(os.cpu_count(), max(1, len(all_folders)))
    print(f"Using up to {max_workers} CPU workers.")

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for folder in all_folders:
            futures.append(executor.submit(process_single_folder, folder))

        for fut in tqdm(as_completed(futures), total=len(futures)):
            folder_path, ok, fail, rows = fut.result()
            total_ok += ok
            total_fail += fail
            total_rows += rows

    print("==========================================")
    print(f"✅ ALL DONE.")
    print(f"Total processed images (OK):   {total_ok}")
    print(f"Total failed images (Fail):    {total_fail}")
    print(f"Total landmark rows in CSVs:   {total_rows}")
    print(f"Outputs (images + CSVs) saved under: {OUTPUT_ROOT}")
💡 ملاحظات مهمّة لك
تشغيل على CPU فقط:

device='cpu' في init_face_alignment

torch.set_num_threads(os.cpu_count()) للاستفادة من تعدد الأنوية

المعالجة بالمجلد (كما طلبتِ):

كل مجلد داخل الـ ZIP →
مجلد مطابق داخل OUTPUT_ROOT

ملف واحد landmarks_68pts.csv

صيغة CSV لكل مجلد:

الأعمدة:

image_name

x_0 .. x_67

y_0 .. y_67

كل صف = صورة واحدة تم اكتشاف الوجه فيها بنجاح

إذا فيه صور بدون وجه:

ما تنحفظ بالصورة ولا تنضاف في CSV (يُعتبر Fail)

إذا حابة:

نضيف أعمدة أكثر في CSV (مثل الـ bbox، أو حجم الوجه، إلخ)

أو نغير طريقة توزيع العمل (بدل مجلد → ملف معين، أو subset بقائمة txt)

قولِي لي فقط وش تبين بالضبط، وأعدّل لك الكود عليه مباشرة 🌷

You said:
ok. one modification. the zip file once extracted it will has folder has the same name of the zip file and inside it there will 100 folders and there is the images. now where i can modify this in the code. 
ChatGPT said:
Perfect, that’s a very clear constraint 👍

Right now the code assumes that the 100 folders are directly under EXTRACT_ROOT.
But in your case the structure is:

EXTRACT_ROOT/
    your_dataset.zip  (not actual file, just name)
    your_dataset/     <-- folder created from zip name
        folder_001/
        folder_002/
        ...
        folder_100/
So we just need to go one level deeper:
EXTRACT_ROOT/zip_folder_name instead of EXTRACT_ROOT.

I’ll show you exactly what to change.

✅ Step 1: Add a DATA_ROOT under the ZIP-name folder
Right after extraction (or after we skip extraction), add this:

# ==========================================
# 1) Extract ZIP (only once)
# ==========================================
if not os.path.isdir(EXTRACT_ROOT) or len(os.listdir(EXTRACT_ROOT)) == 0:
    print(f"Extracting ZIP from: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_ROOT)
    print("✅ Extraction done.")
else:
    print(f"EXTRACT_ROOT already has files, skipping extraction: {EXTRACT_ROOT}")

# >>> ADD THIS BLOCK <<<
# If the ZIP creates a top-level folder with the same name as the zip file,
# we point DATA_ROOT to that folder.
zip_base = os.path.splitext(os.path.basename(ZIP_PATH))[0]
DATA_ROOT = os.path.join(EXTRACT_ROOT, zip_base)
if not os.path.isdir(DATA_ROOT):
    # Fallback: in case structure is different, still use EXTRACT_ROOT
    print(f"[WARN] Expected folder '{zip_base}' inside EXTRACT_ROOT, using EXTRACT_ROOT directly.")
    DATA_ROOT = EXTRACT_ROOT
else:
    print(f"Using DATA_ROOT: {DATA_ROOT}")
✅ Step 2: Change where we collect the 100 folders
Previously we had:

# 2) Collect first-level folders to process
all_folders = sorted(
    p for p in glob.glob(os.path.join(EXTRACT_ROOT, "*"))
    if os.path.isdir(p)
)

print(f"Found {len(all_folders)} folders to process under {EXTRACT_ROOT}.")
👉 Replace that block with this:

# 2) Collect first-level folders to process (under DATA_ROOT now)
all_folders = sorted(
    p for p in glob.glob(os.path.join(DATA_ROOT, "*"))
    if os.path.isdir(p)
)

print(f"Found {len(all_folders)} folders to process under {DATA_ROOT}.")
✅ Step 3: Small adjustment inside process_single_folder
Inside process_single_folder, there was this line:

rel_folder = os.path.relpath(folder_path, EXTRACT_ROOT)
Change it to use DATA_ROOT instead of EXTRACT_ROOT:

rel_folder = os.path.relpath(folder_path, DATA_ROOT)
So that the output structure on Drive mirrors:

DATA_ROOT/folder_001 → OUTPUT_ROOT/folder_001

DATA_ROOT/folder_002 → OUTPUT_ROOT/folder_002

etc.

🔁 Summary of all edits
Define DATA_ROOT after extraction using the zip filename:

zip_base = os.path.splitext(os.path.basename(ZIP_PATH))[0]
DATA_ROOT = os.path.join(EXTRACT_ROOT, zip_base)
...
Use DATA_ROOT when:

Collecting top-level folders (all_folders)

Computing rel_folder inside process_single_folder

If you want, I can paste the full updated script with these changes already integrated so you can copy–paste without touching anything 🌷

You said:
hello again. i want to modify the code. so only i want to preprocess 10 folders i.e., from(10 to 19) folders only.and make the rest the same (# ==========================================
# Simple Processing + ZIP + Multiprocessing + CSV per folder (CPU only)
# ==========================================

!pip -q install face-alignment==1.3.5 opencv-python==4.10.0.84 imageio==2.35.1

import os
import glob
import cv2
import torch
import math
import numpy as np
from tqdm import tqdm
import face_alignment
import zipfile
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# CONFIG (EDIT THESE PATHS)
# ==========================================
# 1) Path to your ZIP file on Drive
ZIP_PATH = "/content/drive/MyDrive/Datasets/DFFDataset/wiki.zip"

# 2) Where to extract the ZIP inside Colab
EXTRACT_ROOT = "/content/extracted_dataset"

# 3) Where to save processed images + CSVs on Drive
OUTPUT_ROOT = "/content/drive/MyDrive/Datasets/Experiments_Preprocessing/SimpleProcessing_wiki"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Padding controls (same as your original code)
MARGIN_X = 1.3
MARGIN_Y = 1.6
SKEW_UP  = 2/3  # 2/3 padding above, 1/3 below

TARGET = 224

# Force CPU (as you requested)
device = 'cpu'
torch.set_num_threads(os.cpu_count())


# ==========================================
# Helper functions
# ==========================================
def clamp_box(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0, min(int(round(xmin)), w - 1))
    ymin = max(0, min(int(round(ymin)), h - 1))
    xmax = max(0, min(int(round(xmax)), w - 1))
    ymax = max(0, min(int(round(ymax)), h - 1))
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)
    return xmin, ymin, xmax, ymax


def largest_face(landmarks_list):
    if len(landmarks_list) == 1:
        return landmarks_list[0]
    best = landmarks_list[0]
    best_area = -1
    for lm in landmarks_list:
        hull = cv2.convexHull(lm.astype(np.float32))
        area = cv2.contourArea(hull)
        if area > best_area:
            best_area = area
            best = lm
    return best


def crop_and_iso_resize_to_224_no_padding(bgr, x1, y1, x2, y2, target=224, interpolation=cv2.INTER_AREA):
    """
    1) Crop bbox.
    2) Isotropic resize so SHORTER side becomes 224 (so both sides >= 224).
    3) Center-crop the longer side to 224 using floor/ceil split (no padding).
    Returns exactly (224,224,3) or None.
    """
    crop = bgr[y1:y2+1, x1:x2+1]
    if crop.size == 0:
        return None

    ch, cw = crop.shape[:2]
    if ch <= 1 or cw <= 1:
        return None

    # Scale by SHORTER side
    scale = float(target) / float(min(ch, cw))
    new_w = int(round(cw * scale))
    new_h = int(round(ch * scale))
    new_w = max(target, new_w)
    new_h = max(target, new_h)

    resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
    H, W = resized.shape[:2]

    # If height is longer than target -> crop rows
    if H > target:
        delta = H - target
        r1 = int(math.floor(delta / 2.0))
        r2 = int(math.ceil(delta / 2.0))
        resized = resized[r1:H - r2, :]
        H, W = resized.shape[:2]

    # If width is longer than target -> crop cols
    if W > target:
        delta = W - target
        c1 = int(math.floor(delta / 2.0))
        c2 = int(math.ceil(delta / 2.0))
        resized = resized[:, c1:W - c2]
        H, W = resized.shape[:2]

    # Safety: enforce exact 224×224 (rare rounding issues)
    if resized.shape[0] != target or resized.shape[1] != target:
        resized = cv2.resize(resized, (target, target), interpolation=interpolation)

    return resized


def init_face_alignment():
    """Create a face-alignment detector on CPU."""
    FA = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device='cpu',
        flip_input=False
    )
    return FA


# ==========================================
# Folder processing function (runs in each process)
# ==========================================
def process_single_folder(folder_path):
    """
    Process all images in one folder:
      - detect face
      - crop + resize
      - save final image
      - collect 68 landmarks per image
      - write ONE CSV file for this folder

    Returns: (folder_path, ok_count, fail_count, num_rows_in_csv)
    """
    FA = init_face_alignment()

    # Compute relative folder name to recreate structure under OUTPUT_ROOT
    rel_folder = os.path.relpath(folder_path, DATA_ROOT)
    out_folder = os.path.join(OUTPUT_ROOT, rel_folder)
    os.makedirs(out_folder, exist_ok=True)

    # CSV path for this folder
    csv_path = os.path.join(out_folder, "landmarks_68pts.csv")

    # Collect images in this folder only (no recursion)
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    image_paths = sorted(
        p for p in glob.glob(os.path.join(folder_path, "*"))
        if os.path.splitext(p)[1].lower() in valid_ext
    )

    ok = 0
    fail = 0
    rows = []

    if not image_paths:
        # Nothing to do for this folder
        return (folder_path, ok, fail, 0)

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        base, _ = os.path.splitext(fname)

        bgr = cv2.imread(img_path)
        if bgr is None:
            fail += 1
            continue

        h, w = bgr.shape[:2]

        # 1) Face detection
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lms_list = FA.get_landmarks_from_image(rgb)
        if not lms_list:
            fail += 1
            continue

        # 2) Choose largest face
        landmarks = largest_face(lms_list)  # (68, 2)

        # 3) Tight bbox from landmarks
        x_min, y_min = float(np.min(landmarks[:, 0])), float(np.min(landmarks[:, 1]))
        x_max, y_max = float(np.max(landmarks[:, 0])), float(np.max(landmarks[:, 1]))

        # 4) Offset/margin bbox
        w_box = x_max - x_min
        h_box = y_max - y_min

        left_extra   = 0.5 * MARGIN_X * w_box
        right_extra  = 0.5 * MARGIN_X * w_box
        top_extra    = SKEW_UP * MARGIN_Y * h_box
        bottom_extra = (1 - SKEW_UP) * MARGIN_Y * h_box

        xp_min = x_min - left_extra
        xp_max = x_max + right_extra
        yp_min = y_min - top_extra
        yp_max = y_max + bottom_extra

        xp_min, yp_min, xp_max, yp_max = clamp_box(xp_min, yp_min, xp_max, yp_max, w, h)

        # 5) Crop extended bbox + isotropic resize (no padding)
        out_img = crop_and_iso_resize_to_224_no_padding(
            bgr, xp_min, yp_min, xp_max, yp_max,
            target=TARGET, interpolation=cv2.INTER_AREA
        )
        if out_img is None:
            fail += 1
            continue

        # 6) Save final image (inside this folder's output)
        out_path = os.path.join(out_folder, f"{base}.jpg")
        saved = cv2.imwrite(out_path, out_img)
        if not saved:
            fail += 1
            continue

        # 7) Collect 68-point landmarks into CSV row
        #    Row format: image_name, x_0 ... x_67, y_0 ... y_67
        xs = landmarks[:, 0].tolist()
        ys = landmarks[:, 1].tolist()
        row = [fname] + xs + ys
        rows.append(row)
        ok += 1

    # Write CSV ONCE per folder
    if rows:
        header = (["image_name"] +
                  [f"x_{i}" for i in range(68)] +
                  [f"y_{i}" for i in range(68)])
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    return (folder_path, ok, fail, len(rows))


# ==========================================
# 1) Extract ZIP (only once)
# ==========================================
if not os.path.isdir(EXTRACT_ROOT) or len(os.listdir(EXTRACT_ROOT)) == 0:
    print(f"Extracting ZIP from: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_ROOT)
    print("✅ Extraction done.")
else:
    print(f"EXTRACT_ROOT already has files, skipping extraction: {EXTRACT_ROOT}")
zip_base = os.path.splitext(os.path.basename(ZIP_PATH))[0]
DATA_ROOT = os.path.join(EXTRACT_ROOT, zip_base)
if not os.path.isdir(DATA_ROOT):
    # Fallback: in case structure is different, still use EXTRACT_ROOT
    print(f"[WARN] Expected folder '{zip_base}' inside EXTRACT_ROOT, using EXTRACT_ROOT directly.")
    DATA_ROOT = EXTRACT_ROOT
else:
    print(f"Using DATA_ROOT: {DATA_ROOT}")

# ==========================================
# 2) Collect first-level folders to process
#    (Assuming your ZIP has ~100 folders at top level)
# ==========================================
# 2) Collect first-level folders to process (under DATA_ROOT now)
all_folders = sorted(
    p for p in glob.glob(os.path.join(DATA_ROOT, "*"))
    if os.path.isdir(p)
)

print(f"Found {len(all_folders)} folders to process under {DATA_ROOT}.")

# ==========================================
# 3) Run multiprocessing over folders (CPU)
# ==========================================
if __name__ == "__main__":
    total_ok = 0
    total_fail = 0
    total_rows = 0

    # You can reduce max_workers (e.g., 2 or 4) if memory is tight
    max_workers = min(os.cpu_count(), max(1, len(all_folders)))
    print(f"Using up to {max_workers} CPU workers.")

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for folder in all_folders:
            futures.append(executor.submit(process_single_folder, folder))

        for fut in tqdm(as_completed(futures), total=len(futures)):
            folder_path, ok, fail, rows = fut.result()
            total_ok += ok
            total_fail += fail
            total_rows += rows

    print("==========================================")
    print(f"✅ ALL DONE.")
    print(f"Total processed images (OK):   {total_ok}")
    print(f"Total failed images (Fail):    {total_fail}")
    print(f"Total landmark rows in CSVs:   {total_rows}")
    print(f"Outputs (images + CSVs) saved under: {OUTPUT_ROOT}")
) 
ChatGPT said:

Thought for 5s
حياك 🌷

تمام، نثبّت اختيار 10 مجلّدات فقط (من 10 إلى 19) ونخلّي الباقي زي ما هو.

الفكرة باختصار:
بدل ما نستخدم all_folders كامل، نختار منه شريحة:

selected_folders = all_folders[10:20]
ملاحظة مهمّة:
هذا السطر في بايثون يعني "من الفهرس 10 إلى أقل من 20" → أي المجلد رقم 11 إلى 20 بالعدّ البشري.
إذا تبينها تكون المجلدات من 10 إلى 19 بالعدّ البشري (10 مجلدات)، استخدمي [9:19] بدل [10:20].

أنا الآن أعدّل لك الجزء السفلي من الكود فقط، والباقي يظل كما هو 👇

🔧 التعديلات المطلوبة في قسم المجلدات + الـ multiprocessing
استبدلي هذا الجزء:

# ==========================================
# 2) Collect first-level folders to process
#    (Assuming your ZIP has ~100 folders at top level)
# ==========================================
# 2) Collect first-level folders to process (under DATA_ROOT now)
all_folders = sorted(
    p for p in glob.glob(os.path.join(DATA_ROOT, "*"))
    if os.path.isdir(p)
)

print(f"Found {len(all_folders)} folders to process under {DATA_ROOT}.")

# ==========================================
# 3) Run multiprocessing over folders (CPU)
# ==========================================
if __name__ == "__main__":
    total_ok = 0
    total_fail = 0
    total_rows = 0

    # You can reduce max_workers (e.g., 2 or 4) if memory is tight
    max_workers = min(os.cpu_count(), max(1, len(all_folders)))
    print(f"Using up to {max_workers} CPU workers.")

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for folder in all_folders:
            futures.append(executor.submit(process_single_folder, folder))

        for fut in tqdm(as_completed(futures), total=len(futures)):
            folder_path, ok, fail, rows = fut.result()
            total_ok += ok
            total_fail += fail
            total_rows += rows

    print("==========================================")
    print(f"✅ ALL DONE.")
    print(f"Total processed images (OK):   {total_ok}")
    print(f"Total failed images (Fail):    {total_fail}")
    print(f"Total landmark rows in CSVs:   {total_rows}")
    print(f"Outputs (images + CSVs) saved under: {OUTPUT_ROOT}")
بهذا الإصدار المعدّل الذي يعالج فقط المجلدات من 10 إلى 19:

# ==========================================
# 2) Collect first-level folders to process (under DATA_ROOT now)
# ==========================================
all_folders = sorted(
    p for p in glob.glob(os.path.join(DATA_ROOT, "*"))
    if os.path.isdir(p)
)

print(f"Found {len(all_folders)} folders under {DATA_ROOT}.")

# ✅ اختاري فقط المجلدات من الفهرس 10 إلى 19 (10 مجلدات)
# هذا يعني: folders[10], folders[11], ..., folders[19]
selected_folders = all_folders[10:20]

print(f"Will process ONLY {len(selected_folders)} folders (indexes 10 to 19 in the sorted list).")

# ==========================================
# 3) Run multiprocessing over ONLY the selected folders (CPU)
# ==========================================
if __name__ == "__main__":
    total_ok = 0
    total_fail = 0
    total_rows = 0

    # You can reduce max_workers (e.g., 2 or 4) if memory is tight
    max_workers = min(os.cpu_count(), max(1, len(selected_folders)))
    print(f"Using up to {max_workers} CPU workers.")

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for folder in selected_folders:
            futures.append(executor.submit(process_single_folder, folder))

        for fut in tqdm(as_completed(futures), total=len(futures)):
            folder_path, ok, fail, rows = fut.result()
            total_ok += ok
            total_fail += fail
            total_rows += rows

    print("==========================================")
    print(f"✅ ALL DONE (PARTIAL RUN: folders 10–19 only).")
    print(f"Total processed images (OK):   {total_ok}")
    print(f"Total failed images (Fail):    {total_fail}")
    print(f"Total landmark rows in CSVs:   {total_rows}")
    print(f"Outputs (images + CSVs) saved under: {OUTPUT_ROOT}")
لو حابة تتحكّمين بمدى أبسط
ممكن حتى تضيفي متغيّرات في الأعلى:

START_IDX = 10
END_IDX   = 20   # غير شاملة، زي slicing في بايثون

selected_folders = all_folders[START_IDX:END_IDX]
فتقدرين تغيّرين النطاق بسهولة من مكان واحد بدون تعديل أكثر من سطر.

لو تبين بعدها نسوي نسخة ثانية تعالج باقي المجلدات (0–9، أو 20–آخر شيء) أقدر أرتب لك سكريبت ثاني يكمل الباقي بدون لمس المجلدات اللي خلصتِها 🌸

You said:
hello, can you regenerate for me my question and answer on last wensedy?i wrote to you a code and i want it
ChatGPT said:

Thought for 8s
Right now I don’t have a full “time machine” into all of our old chats, so I can’t reliably reconstruct the exact question + answer from last Wednesday the way you’re imagining 😔

I only see short summaries/snippets of older conversations, not the complete messages or every code cell we exchanged. From last Wednesday (2026-01-28) the logs I see look like:

A chat about the DRN GitHub repo: you pasted the start of classify.py:

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

import drn as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def parse_args():
    ...
And we were debugging classification / dataset folders for DRN.

But the exact code I wrote back for you (the full modified script) is not fully visible to me now – it’s inside the “conversation too long; truncated” parts, which I can’t reopen.

What we can do now
I can’t 1:1 recover that old answer, but I can:

Re-write the same type of code for you if you describe:

roughly what that script did (e.g., “DRN training script with fixed test folder + accuracy print” / “preprocessing pipeline with canonical face alignment” / etc.), or

which part you remember (e.g., some function names, variables, or key behavior).

Or if you know which file it was about (DRN classify, preprocessing from ZIP, canonical face centering, etc.), I can regenerate a clean full version of that script for you now.

To make this concrete, you can say something like:

“It was the code that did X (e.g. DRN training on 0_real/1_fake with fixed testing\crn structure and printed best accuracy).”

and I’ll rebuild that script fully for you in this chat so you can copy–paste it into Colab.

You said:
ok. please make the necessery modification for the following code(import argparse
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
    Wraps a base DRN model and applies your HFRF block
    (HFRI + conv + HFRFC) on the INPUT IMAGE before DRN.

    Input:  x (B, 3, H, W)  # normalized image from DataLoader
    Output: base_model(x_hfrf)
    """

    def __init__(self, base_model, scale_wh=4, scale_c=4):
        super(HFRFWrapper, self).__init__()
        self.base_model = base_model
        self.scale_wh = scale_wh
        self.scale_c = scale_c

        # Small conv layer that corresponds to:
        # "Apply a convolution layer on the output of (HFRI) block"
        # We keep 3->3 so DRN still gets 3-channel input.
        self.pre_conv = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        nn.init.kaiming_normal_(self.pre_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.pre_conv.bias is not None:
            nn.init.constant_(self.pre_conv.bias, 0.0)

    # ---------- HFRI: high-frequency over spatial dims (W,H) ----------
    def hfreqWH(self, x, scale: int):
        """
        x: (B, C, H, W)
        High-pass filter in spatial frequency domain.
        """
        assert scale > 2, "scale must be > 2"

        # 2D FFT over H,W
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])

        b, c, h, w = x.shape

        # zero out LOW-frequency center block
        x[:, :,
          h // 2 - h // scale : h // 2 + h // scale,
          w // 2 - w // scale : w // 2 + w // scale] = 0.0

        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")

        x = torch.real(x)
        x = F.relu(x, inplace=True)
        return x

    # ---------- HFRFC: high-frequency along channel dimension C ----------
    def hfreqC(self, x, scale: int):
        """
        x: (B, C, H, W)
        High-pass filter in frequency domain along channel axis.
        """
        assert scale > 2, "scale must be > 2"

        # 1D FFT along channel dimension
        x = torch.fft.fft(x, dim=1, norm="ortho")
        x = torch.fft.fftshift(x, dim=1)

        b, c, h, w = x.shape

        # zero out LOW-frequency center block in channel spectrum
        x[:, c // 2 - c // scale : c // 2 + c // scale, :, :] = 0.0

        x = torch.fft.ifftshift(x, dim=1)
        x = torch.fft.ifft(x, dim=1, norm="ortho")

        x = torch.real(x)
        x = F.relu(x, inplace=True)
        return x

    def forward(self, x):
        # Just to debug shapes like in your snippet:
        print(f'output shape of input image: {x.shape}')

        # === HFRI on the raw image ===
        x = self.hfreqWH(x, self.scale_wh)

        # === Conv layer on HFRI output ===
        # equivalent to your: F.conv2d(x, self.weight1, self.bias1, ...)
        x = self.pre_conv(x)
        x = F.relu(x, inplace=True)

        # === HFRFC on the conv feature map ===
        x = self.hfreqC(x, self.scale_c)

        # === Now pass to the original DRN model ===
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

    # ----- Wrap with HFRF block -----
    # This will apply your FFT-based high-frequency block on the input image
    model = HFRFWrapper(base_model, scale_wh=4, scale_c=4)

    # ----- NOW move to GPU & DataParallel -----
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
    base_model.fc = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1, stride=1, padding=0, bias=True)
    nn.init.kaiming_normal_(base_model.fc.weight, mode='fan_out', nonlinearity='relu')
    if base_model.fc.bias is not None:
        nn.init.constant_(base_model.fc.bias, 0.)

    # wrap with HFRF block
    model = HFRFWrapper(base_model, scale_wh=4, scale_c=4)

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
    main()). note that i want you to delete the frequency functions part and make the code only classify the input images with DRN classifier noting extra.
ChatGPT said:

Thought for 5s
Got it 🙌
We’ll strip out all the FFT / HFRF stuff and use plain DRN as a binary classifier. No extra preprocessing in the model, just DRN + your training/test/eval logic.

What I changed
❌ Removed HFRFWrapper class completely (and all hfreqWH, hfreqC, FFT, etc.).

❌ Removed import torch.nn.functional as F (no longer needed).

✅ In run_training: create a plain DRN model, change its fc to 2 classes, then wrap with DataParallel and move to CUDA.

✅ In test_model: do the same (plain DRN, adjusted fc, then load checkpoints and evaluate).

✅ Kept:

training loop

validation loop

metrics (accuracy, sensitivity, specificity, etc.)

checkpoint saving & evaluation of LAST and BEST models.

Here’s the clean, modified full script:

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


def run_training(args):
    # ----- create base DRN model -----
    model = models.__dict__[args.arch](args.pretrained)

    # ----- adjust classifier to 2 classes -----
    NUM_CLASSES = 2
    in_channels = model.out_dim          # DRN defines this as last feature dim
    model.fc = nn.Conv2d(
        in_channels,
        NUM_CLASSES,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True
    )
    nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    if model.fc.bias is not None:
        nn.init.constant_(model.fc.bias, 0.)

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
    model = models.__dict__[args.arch](args.pretrained)

    # adjust classifier to 2 classes
    NUM_CLASSES = 2
    in_channels = model.out_dim
    model.fc = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1, stride=1, padding=0, bias=True)
    nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    if model.fc.bias is not None:
        nn.init.constant_(model.fc.bias, 0.)

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

    # Data loading code
    # NOTE: your testing path: data/testing/crn
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
If you later decide you want only pure classification (no sensitivity/specificity stuff, no LAST/BEST comparison), I can also give you a minimal clean version of test_model and remove the extra evaluation logic.

You said:
ok thanks. but i want to add modification also. the input images is on a zip folder. so, in the code i want you to unzip the files then start to read the images. 
ChatGPT said:

Thought for 8s
Nice, that’s a clean modification to add 👍
We’ll make your script accept either:

a normal folder (with training/, validation/, testing/ inside), or

a ZIP file that contains those folders → we unzip it once, then train/test from the extracted path.

What I’m adding
import zipfile

Helper function prepare_data_root(data_path) that:

if data_path is a directory → just returns it

if it’s a *.zip file → extracts it to a folder next to the zip (e.g. .../wiki_extracted) and returns that

In run_training and test_model:

use data_root = prepare_data_root(args.data)

replace os.path.join(args.data, ...) with os.path.join(data_root, ...)

You can now call, for example:

python classify.py train /content/drive/MyDrive/Datasets/DFFDataset/wiki.zip
python classify.py test  /content/drive/MyDrive/Datasets/DFFDataset/wiki.zip
Full modified script (DRN only + ZIP support)
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

import zipfile  # <- NEW

import drn as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test', 'map', 'locate'])
    parser.add_argument('data', metavar='DIR_OR_ZIP',
                        help='path to dataset folder OR a .zip file')
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
    If data_path is a directory: return it.
    If data_path is a .zip file: extract it (once) and return the extraction folder.
    Expected inside: training/, validation/, testing/...
    """
    if os.path.isdir(data_path):
        print(f"[INFO] Using existing directory as data root: {data_path}")
        return data_path

    if os.path.isfile(data_path) and data_path.lower().endswith('.zip'):
        base_dir = os.path.dirname(os.path.abspath(data_path))
        zip_base = os.path.splitext(os.path.basename(data_path))[0]
        extract_root = os.path.join(base_dir, zip_base + "_extracted")

        if not os.path.isdir(extract_root) or len(os.listdir(extract_root)) == 0:
            print(f"[INFO] Extracting ZIP dataset from: {data_path}")
            os.makedirs(extract_root, exist_ok=True)
            with zipfile.ZipFile(data_path, 'r') as zf:
                zf.extractall(extract_root)
            print(f"[INFO] Extraction done. Data root: {extract_root}")
        else:
            print(f"[INFO] Using already extracted data root: {extract_root}")

        # If the zip created an extra top-level folder, try to detect training/validation/testing.
        # Simple heuristic: if extract_root directly has 'training', use it; else search one level down.
        has_training = os.path.isdir(os.path.join(extract_root, "training"))
        if has_training:
            return extract_root

        # Search one level deep for a folder that contains 'training'
        for entry in os.listdir(extract_root):
            candidate = os.path.join(extract_root, entry)
            if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "training")):
                print(f"[INFO] Found split folders under: {candidate}")
                return candidate

        # Fallback: return extract_root anyway (user can organize as needed)
        print(f"[WARN] 'training' folder not found; using {extract_root} as data root.")
        return extract_root

    raise ValueError(f"Data path '{data_path}' is neither a directory nor a .zip file.")


# ==========================
# TRAINING
# ==========================
def run_training(args):
    # ----- prepare data root (folder or zip) -----
    data_root = prepare_data_root(args.data)

    # ----- create base DRN model -----
    model = models.__dict__[args.arch](args.pretrained)

    # ----- adjust classifier to 2 classes -----
    NUM_CLASSES = 2
    in_channels = model.out_dim          # DRN defines this as last feature dim
    model.fc = nn.Conv2d(
        in_channels,
        NUM_CLASSES,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True
    )
    nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    if model.fc.bias is not None:
        nn.init.constant_(model.fc.bias, 0.)

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


# ==========================
# TESTING
# ==========================
def test_model(args):
    # ----- prepare data root (folder or zip) -----
    data_root = prepare_data_root(args.data)

    # base DRN model
    model = models.__dict__[args.arch](args.pretrained)

    # adjust classifier to 2 classes
    NUM_CLASSES = 2
    in_channels = model.out_dim
    model.fc = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1, stride=1, padding=0, bias=True)
    nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    if model.fc.bias is not None:
        nn.init.constant_(model.fc.bias, 0.)

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

    # Data loading code
    # NOTE: your testing path: data/testing/crn
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


# ==========================
# TRAIN / VAL LOOP + METRICS
# ==========================
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
If you tell me the exact internal structure of your zip (e.g. does it have an extra wiki/ folder then training/validation/testing?), I can tune prepare_data_root specifically for that so it always finds the right split directories without you touching anything.

You said:
no, the folder of all dataset is a zip file. no neeed to exract the train and test and val. only extract the big or out folder and the rest will come on normaaly. please adjust.
ChatGPT said:

Thought for 6s
اوكي تمام، فهمتك 👍

يعني الـ ZIP فيه مجلّد رئيسي واحد فقط (مثلاً wiki/) وداخله training/, validation/, testing/…
أنتِ تبين:

نفك الضغط مرة واحدة بس للمجلد الكبير

وبعدين الكود يتعامل عادي مع training/, validation/, testing/ بدون لف ودوران أو بحث إضافي

فأنا عدّلت دالة prepare_data_root بحيث:

إذا args.data = مسار مجلد → يرجّعه مباشرة

إذا .zip:

يفكّه في مجلد اسمه <zip_name>_extracted

يحدّد data_root = <extract_root>/<zip_name> (المجلد الكبير داخل الـ zip)

لو هذا المجلد غير موجود، يرجع extract_root مع تحذير

باقي الكود كما هو (DRN فقط، تصنيف ثنائي، نفس الـ train/test/eval).

النسخة المعدّلة بالكامل (مع فك الـ ZIP للمجلد الرئيسي فقط)
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

import zipfile  # <- نستخدمه لفك الـ ZIP

import drn as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test', 'map', 'locate'])
    parser.add_argument('data', metavar='DIR_OR_ZIP',
                        help='path to dataset folder OR a .zip file')
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
    إذا كان data_path مجلد: يرجّعه كما هو.
    إذا كان data_path ملف .zip:
      - نفكّه في مجلد <zip_name>_extracted
      - نعتبر data_root = <extract_root>/<zip_name> (المجلد الكبير داخل الـ zip)
    الترتيب المتوقع داخل data_root:
        training/
        validation/
        testing/
        ...
    """
    if os.path.isdir(data_path):
        print(f"[INFO] Using existing directory as data root: {data_path}")
        return data_path

    if os.path.isfile(data_path) and data_path.lower().endswith('.zip'):
        base_dir = os.path.dirname(os.path.abspath(data_path))
        zip_base = os.path.splitext(os.path.basename(data_path))[0]

        # نفك الضغط في مجلد منفصل
        extract_root = os.path.join(base_dir, zip_base + "_extracted")

        if not os.path.isdir(extract_root) or len(os.listdir(extract_root)) == 0:
            print(f"[INFO] Extracting ZIP dataset from: {data_path}")
            os.makedirs(extract_root, exist_ok=True)
            with zipfile.ZipFile(data_path, 'r') as zf:
                zf.extractall(extract_root)
            print(f"[INFO] Extraction done under: {extract_root}")
        else:
            print(f"[INFO] Using already extracted data under: {extract_root}")

        # نفترض أن داخل extract_root يوجد مجلد اسمه zip_base
        # هو المجلد الكبير اللي فيه training/validation/testing
        data_root = os.path.join(extract_root, zip_base)
        if os.path.isdir(data_root):
            print(f"[INFO] Using data root: {data_root}")
            return data_root
        else:
            # لو ما لقينا المجلد بهذا الاسم، نرجع extract_root مع تحذير
            print(f"[WARN] Expected folder '{zip_base}' inside {extract_root}, "
                  f"but not found. Using {extract_root} as data root.")
            return extract_root

    raise ValueError(f"Data path '{data_path}' is neither a directory nor a .zip file.")


# ==========================
# TRAINING
# ==========================
def run_training(args):
    # ----- prepare data root (folder or zip) -----
    data_root = prepare_data_root(args.data)

    # ----- create base DRN model -----
    model = models.__dict__[args.arch](args.pretrained)

    # ----- adjust classifier to 2 classes -----
    NUM_CLASSES = 2
    in_channels = model.out_dim          # DRN defines this as last feature dim
    model.fc = nn.Conv2d(
        in_channels,
        NUM_CLASSES,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True
    )
    nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    if model.fc.bias is not None:
        nn.init.constant_(model.fc.bias, 0.)

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


# ==========================
# TESTING
# ==========================
def test_model(args):
    # ----- prepare data root (folder or zip) -----
    data_root = prepare_data_root(args.data)

    # base DRN model
    model = models.__dict__[args.arch](args.pretrained)

    # adjust classifier to 2 classes
    NUM_CLASSES = 2
    in_channels = model.out_dim
    model.fc = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1, stride=1, padding=0, bias=True)
    nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    if model.fc.bias is not None:
        nn.init.constant_(model.fc.bias, 0.)

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


# ==========================
# TRAIN / VAL LOOP + METRICS
# ==========================
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
لو حابّة أتأكد معك من شكل الـ ZIP (مثلاً: wiki.zip → wiki/training/...) اكتبيه لي نصًّا، بس من اللي قلتيه هذا الكود يناسب وضعك بالضبط: يفك المجلد الكبير مرة واحدة والباقي يشتغل عادي 🎯






No file chosenNo file chosen
ChatGPT can make mistakes. Check important info.
