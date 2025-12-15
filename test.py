import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import warnings

import numpy as np
import pandas as pd
import imageio.v3 as iio
import pydicom
from PIL import Image
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from transformers import AutoProcessor, SiglipProcessor

from config import CONFIG
from model import PromptedMultiScaleMedSigLIPIQA

warnings.filterwarnings("ignore")

# -------------------- DEVICE --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device}")

# -------------------- CHECKPOINT PATH --------------------
CHECKPOINT_PATH = "./checkpoints/medsiglip_iqa_model_best.pth"


# -------------------- HELPER FUNCTIONS --------------------
def build_processor(name):
    try:
        proc = AutoProcessor.from_pretrained(name)
        print(f"✅ Processor loaded: {name}")
        return proc
    except Exception as e:
        print(f"❌ AutoProcessor could not be loaded ({e}).")
        print("↪️ Falling back to SiglipProcessor: google/siglip-base-patch16-224")
        return SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")


def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan


def index_files(root, exts={".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}):
    idx = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            full = os.path.join(dp, fn)
            base = os.path.basename(fn)
            lowb = base.lower()
            stem = os.path.splitext(lowb)[0]
            ext = os.path.splitext(lowb)[1]
            if exts and ext not in exts:
                continue
            idx[lowb] = full
            idx[stem] = full
    return idx


def load_test_list(json_path, data_dir):
    # Load ground truth data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Index all files in the directory for fuzzy matching
    file_index = index_files(data_dir)
    rows, missing = [], []
    for k, v in data.items():
        if v is None:
            continue
        k = str(k)
        full = os.path.join(data_dir, k)

        # Direct match
        if os.path.exists(full):
            rows.append((full, safe_float(v)))
            continue

        # Fuzzy match (case insensitive, extension agnostic)
        candidates = [
            k.lower(),
            os.path.basename(k).lower(),
            os.path.splitext(k.lower())[0],
            os.path.splitext(os.path.basename(k).lower())[0],
        ]
        found = None
        for c in candidates:
            if c in file_index:
                found = file_index[c]
                break
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"]:
                if (c + ext) in file_index:
                    found = file_index[c + ext]
                    break
            if found:
                break

        if found:
            rows.append((found, safe_float(v)))
        else:
            missing.append(k)

    df = pd.DataFrame(rows, columns=["path", "mos"]).dropna(subset=["mos"])
    print(f"Test images: {len(df)} | Missing: {len(missing)}")
    if missing[:5]:
        print("Sample missing:", missing[:5])
    return df


def read_image(path):
    ext = os.path.splitext(path)[1].lower()
    # Handle DICOM files
    if ext == ".dcm":
        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            # Normalize DICOM to 0-255
            img = img - img.min()
            if img.max() > 0:
                img = img / img.max()
            img = (img * 255).astype(np.uint8)
            if img.ndim == 2:
                img = np.stack([img, img, img], -1)
            return img
        except Exception as e:
            print(f"DICOM read error {path}: {e}")
    # Handle standard image formats
    return iio.imread(path)


# -------------------- DATASET --------------------
class IQATestDataset(Dataset):
    def __init__(
        self, df, processor, image_size=448, transform=None, return_path=False
    ):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.image_size = image_size
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.loc[idx, "path"]
        y = float(self.df.loc[idx, "mos"])

        try:
            img = read_image(p)
        except Exception as e:
            print(f"Read error {p}: {e}")
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Dimension checks
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[-1] > 3:
            img = img[..., :3]

        # Dtype conversion
        if img.dtype in [np.float32, np.float64]:
            # Empty image check
            if img.size == 0 or (img.max() == 0 and img.min() == 0):
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            elif img.max() <= 1.0 and img.min() >= 0.0:
                img = (img * 255).astype(np.uint8)
            else:
                # Normalize and clip
                img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            # Overflow check for integer types
            img = np.clip(img, 0, 255).astype(np.uint8)

        pil = Image.fromarray(img)

        # Apply transform
        if self.transform is not None:
            pil = self.transform(pil)
        else:
            pil = pil.resize(
                (self.image_size, self.image_size), Image.Resampling.LANCZOS
            )

        # Process with processor
        try:
            inputs = self.processor(
                images=pil, return_tensors="pt", do_resize=False, do_center_crop=False
            )
        except TypeError:
            inputs = self.processor(images=pil, return_tensors="pt")
        except Exception as e:
            print(f"Processor error: {e} — manual normalization.")
            tr = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            inputs = {"pixel_values": tr(pil).unsqueeze(0)}

        # Remove batch dimension
        for k in list(inputs.keys()):
            v = inputs[k]
            if hasattr(v, "dim") and v.dim() > 1:
                inputs[k] = v.squeeze(0)

        y = torch.tensor(y, dtype=torch.float32)
        return (inputs, y, p) if self.return_path else (inputs, y)


# Deterministic val/test transform
val_test_transform = T.Compose([T.Resize((CONFIG["image_size"], CONFIG["image_size"]))])

# -------------------- PROCESSOR / MODEL --------------------
processor = build_processor(CONFIG["model_name"])

text_inputs = None
if CONFIG["use_text_prompt"]:
    try:
        ti = processor(text=CONFIG["text_prompt"], return_tensors="pt")
        text_inputs = {k: v for k, v in ti.items() if isinstance(v, torch.Tensor)}
    except Exception as e:
        print("Text processor error:", e)
        text_inputs = None

model = PromptedMultiScaleMedSigLIPIQA(
    CONFIG["model_name"],
    text_inputs=text_inputs,
    film_strength=CONFIG["film_strength"],
    bound=CONFIG["bound_output_0_4"],
).to(device)

state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(state, strict=False)
model.eval()
print(f"✅ Model loaded from: {CHECKPOINT_PATH}")

# -------------------- TEST DATA --------------------
df_test = load_test_list(CONFIG["test_json_path"], CONFIG["test_data_dir"])
if len(df_test) == 0:
    raise SystemExit("❌ Test list is empty. Check paths and test.json content.")

test_ds = IQATestDataset(
    df_test,
    processor,
    image_size=CONFIG["image_size"],
    transform=val_test_transform,
    return_path=False,
)
test_dl = DataLoader(
    test_ds,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
)

# -------------------- PREDICTION + METRICS --------------------
all_preds, all_tgts = [], []
with torch.no_grad():
    for inputs, y in tqdm(test_dl, desc="Test"):
        pv = inputs["pixel_values"].to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        yhat = model({"pixel_values": pv})
        all_preds.extend(yhat.detach().cpu().numpy().tolist())
        all_tgts.extend(y.detach().cpu().numpy().tolist())

all_preds = np.asarray(all_preds, dtype=np.float64)
all_tgts = np.asarray(all_tgts, dtype=np.float64)

results = {
    "plcc": abs(pearsonr(all_preds, all_tgts)[0]),
    "srocc": abs(spearmanr(all_preds, all_tgts)[0]),
    "krocc": abs(kendalltau(all_preds, all_tgts)[0]),
}
results["overall"] = results["plcc"] + results["srocc"] + results["krocc"]

print("\n📊 TEST RESULTS")
print(f"PLCC  : {results['plcc']:.4f}")
print(f"SROCC : {results['srocc']:.4f}")
print(f"KROCC : {results['krocc']:.4f}")
print(f"TOTAL : {results['overall']:.4f}")
