"""CLIP ViT-B/16 baseline at 448x448: test-set evaluation of the five fold checkpoints (PLCC, SROCC, KROCC).

Paths: data/test/test.json and data/test/images by default (override with LDCT_TEST_JSON, LDCT_TEST_DIR);
checkpoints are read from checkpoints/clip_vitb16_448 (override with CLIP_CKPT_DIR).
Extra dependencies: pip install open_clip_torch timm
"""
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
import imageio.v3 as iio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm import tqdm
import torchvision.transforms as T

# -------------------- CONFIG --------------------
CONFIG = {
    "checkpoint_dir":  os.environ.get("CLIP_CKPT_DIR", "checkpoints/clip_vitb16_448"),
    "checkpoint_stem": "clip_vitb16_448_fold",
    "num_folds":       5,
    "test_json_path":  os.environ.get("LDCT_TEST_JSON", "data/test/test.json"),
    "test_data_dir":   os.environ.get("LDCT_TEST_DIR", "data/test/images"),
    "image_size":      448,
    "batch_size":      4,
    "num_workers":     0 if os.name == "nt" else 2,
    "text_prompt": (
        "Rate this low-dose CT (MOS 0-4): 0 Nondiagnostic\u2014desired features not shown; "
        "1 Poor\u2014diagnostic interpretation impossible; 2 Fair\u2014limited interpretation; "
        "3 Good\u2014diagnostic; 4 Excellent\u2014anatomy highly visible. Return only one number 0-4."
    ),
    "film_strength": 1.0,
    "sigmoid_tau":   2.0,
}

BIOMEDCLIP_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMED_MEAN = [0.48145466, 0.4578275, 0.40821073]
BIOMED_STD  = [0.26862954, 0.26130258, 0.27577711]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# -------------------- HELPERS --------------------
def safe_float(x):
    try: return float(x)
    except: return np.nan

def index_files(root, exts={".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}):
    idx = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            full = os.path.join(dp, fn)
            lowb = os.path.basename(fn).lower()
            stem = os.path.splitext(lowb)[0]
            ext  = os.path.splitext(lowb)[1]
            if exts and ext not in exts:
                continue
            idx[lowb] = full
            idx[stem] = full
    return idx

def load_test_list(json_path, data_dir):
    with open(json_path, "r") as f:
        data = json.load(f)
    file_index = index_files(data_dir)
    rows, missing = [], []
    for k, v in data.items():
        if v is None: continue
        k = str(k)
        full = os.path.join(data_dir, k)
        if os.path.exists(full):
            rows.append((full, safe_float(v))); continue
        candidates = [k.lower(), os.path.basename(k).lower(),
                      os.path.splitext(k.lower())[0],
                      os.path.splitext(os.path.basename(k).lower())[0]]
        found = None
        for c in candidates:
            if c in file_index:
                found = file_index[c]; break
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"]:
                if (c + ext) in file_index:
                    found = file_index[c + ext]; break
            if found: break
        if found: rows.append((found, safe_float(v)))
        else: missing.append(k)
    df = pd.DataFrame(rows, columns=["path", "mos"]).dropna(subset=["mos"])
    print(f"Test images: {len(df)} | Missing: {len(missing)}")
    if missing[:5]: print("Sample missing:", missing[:5])
    return df

def read_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        try:
            import pydicom
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            img = img - img.min()
            if img.max() > 0: img /= img.max()
            img = (img * 255).astype(np.uint8)
            if img.ndim == 2: img = np.stack([img, img, img], -1)
            return img
        except Exception as e:
            print(f"DICOM error {path}: {e}")
    return iio.imread(path)

# -------------------- DATASET --------------------
val_transform = T.Compose([
    T.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    T.ToTensor(),
    T.Normalize(mean=BIOMED_MEAN, std=BIOMED_STD),
])

class BiomedCLIPTestDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        p = self.df.loc[idx, "path"]
        y = float(self.df.loc[idx, "mos"])
        try:
            img = read_image(p)
        except Exception as e:
            print(f"Read error {p}: {e}")
            img = np.zeros((CONFIG["image_size"], CONFIG["image_size"], 3), dtype=np.uint8)

        if img.ndim == 3 and img.shape[-1] == 1: img = img.squeeze(-1)
        if img.ndim == 2: img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[-1] > 3: img = img[..., :3]
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        pil = Image.fromarray(img)
        pixel_values = val_transform(pil)
        return {"pixel_values": pixel_values}, torch.tensor(y, dtype=torch.float32)

# -------------------- MODEL --------------------
class BiomedCLIPFiLMIQA(nn.Module):
    def __init__(self, film_strength=1.0, sigmoid_tau=2.0):
        super().__init__()
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-16")

        import timm
        self.visual = nn.Module()
        self.visual.trunk = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True, num_classes=0, img_size=448)
        h = 768   # BiomedCLIP ViT-B/16 trunk embed_dim
        self.hidden_size = h
        self.film_strength = film_strength
        self.tau = sigmoid_tau

        text_dim = 512  # encode_text output dim
        self.text_to_film = nn.Sequential(
            nn.Linear(text_dim, 2 * h), nn.ReLU(),
            nn.Linear(2 * h, 2 * h)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_pool  = nn.AdaptiveAvgPool1d(4)
        self.text_pool   = nn.AdaptiveMaxPool1d(2)

        self.g_head = nn.Sequential(nn.Linear(h,       256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.l_head = nn.Sequential(nn.Linear(h * 4,   256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.t_head = nn.Sequential(nn.Linear(h * 2,   128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
        self.fuse   = nn.Sequential(nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        self.prompt_embed = nn.Parameter(torch.zeros(1, text_dim))
        self._init_prompt_embed(clip_model, tokenizer, CONFIG["text_prompt"])
        del clip_model, tokenizer

    @torch.no_grad()
    def _init_prompt_embed(self, clip_model, tokenizer, prompt: str):
        dev = next(self.visual.parameters()).device
        tokens = tokenizer([prompt]).to(dev)
        pe = clip_model.encode_text(tokens)
        pe = F.normalize(pe, dim=-1)
        self.prompt_embed.data.copy_(pe.detach())

    def forward(self, inputs):
        x = inputs["pixel_values"].to(next(self.visual.parameters()).device)
        all_tokens = self.visual.trunk.forward_features(x)
        patches = all_tokens[:, 1:]

        B, P, H = patches.shape
        pe = self.prompt_embed.expand(B, -1)
        gamma, beta = self.text_to_film(pe).chunk(2, dim=-1)
        patches = patches * (1 + self.film_strength * torch.tanh(gamma).unsqueeze(1)) \
                + self.film_strength * beta.unsqueeze(1)

        patches = patches.transpose(1, 2)
        g = self.global_pool(patches).squeeze(-1)
        l = self.local_pool(patches).flatten(1)
        t = self.text_pool(patches).flatten(1)

        logit = self.fuse(torch.cat([self.g_head(g), self.l_head(l), self.t_head(t)], dim=1)).squeeze(-1)
        return 4.0 * torch.sigmoid(logit / self.tau)

# -------------------- LOAD DATA + MODEL --------------------
df_test = load_test_list(CONFIG["test_json_path"], CONFIG["test_data_dir"])
if len(df_test) == 0:
    raise SystemExit("Test list is empty. Check paths.")

test_ds = BiomedCLIPTestDataset(df_test)
test_dl = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                     num_workers=CONFIG["num_workers"], pin_memory=True)

model = BiomedCLIPFiLMIQA(
    film_strength=CONFIG["film_strength"],
    sigmoid_tau=CONFIG["sigmoid_tau"]
).to(device)

# -------------------- TEST EACH FOLD --------------------
fold_results = []

for fold in range(1, CONFIG["num_folds"] + 1):
    ckpt_path = os.path.join(CONFIG["checkpoint_dir"], f"{CONFIG['checkpoint_stem']}{fold}.pth")
    if not os.path.exists(ckpt_path):
        print(f"Fold {fold} checkpoint not found: {ckpt_path} -- skipping.")
        continue

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(
        state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state,
        strict=False
    )
    model.train(False)

    all_preds, all_tgts = [], []
    with torch.no_grad():
        for inputs, y in tqdm(test_dl, desc=f"Fold {fold}", leave=False):
            pv = inputs["pixel_values"].to(device, non_blocking=True)
            yhat = model({"pixel_values": pv})
            all_preds.extend(yhat.detach().cpu().numpy().tolist())
            all_tgts.extend(y.numpy().tolist())

    preds = np.asarray(all_preds, dtype=np.float64)
    tgts  = np.asarray(all_tgts,  dtype=np.float64)
    plcc  = abs(pearsonr(preds,  tgts)[0])
    srocc = abs(spearmanr(preds, tgts)[0])
    krocc = abs(kendalltau(preds, tgts)[0])
    fold_results.append({"fold": fold, "plcc": plcc, "srocc": srocc,
                         "krocc": krocc, "overall": plcc + srocc + krocc})

# -------------------- RESULTS --------------------
if not fold_results:
    raise SystemExit("No folds could be evaluated.")

plcc_arr    = np.array([r["plcc"]    for r in fold_results])
srocc_arr   = np.array([r["srocc"]   for r in fold_results])
krocc_arr   = np.array([r["krocc"]   for r in fold_results])
overall_arr = np.array([r["overall"] for r in fold_results])

print("\n=== CLIP ViT-B/16 @448 + FiLM: TEST RESULTS ===")
print(f"{'Fold':<6} {'PLCC':>8} {'SROCC':>8} {'KROCC':>8} {'Overall':>10}")
print("-" * 44)
for r in fold_results:
    print(f"Fold {r['fold']:<2} {r['plcc']:>8.4f} {r['srocc']:>8.4f} {r['krocc']:>8.4f} {r['overall']:>10.4f}")
print("-" * 44)
print(f"{'Mean':<6} {plcc_arr.mean():>8.4f} {srocc_arr.mean():>8.4f} {krocc_arr.mean():>8.4f} {overall_arr.mean():>10.4f}")
print(f"{'Std':<6} {plcc_arr.std():>8.4f} {srocc_arr.std():>8.4f} {krocc_arr.std():>8.4f} {overall_arr.std():>10.4f}")
print(f"\nMean +- Std:")
print(f"  PLCC    : {plcc_arr.mean():.4f} +- {plcc_arr.std():.4f}")
print(f"  SROCC   : {srocc_arr.mean():.4f} +- {srocc_arr.std():.4f}")
print(f"  KROCC   : {krocc_arr.mean():.4f} +- {krocc_arr.std():.4f}")
print(f"  Overall : {overall_arr.mean():.4f} +- {overall_arr.std():.4f}")




# --- AUTO-GENERATED TEST RESULTS LOGGER ---
try:
    import json
    results_dict = {
        "folds": fold_results,
        "mean_metrics": {
            "plcc": float(plcc_arr.mean()), 
            "srocc": float(srocc_arr.mean()),
            "krocc": float(krocc_arr.mean()), 
            "overall": float(overall_arr.mean())
        },
        "std_metrics": {
            "plcc": float(plcc_arr.std()), 
            "srocc": float(srocc_arr.std()),
            "krocc": float(krocc_arr.std()), 
            "overall": float(overall_arr.std())
        }
    }
    out_path = os.path.join(CONFIG.get("checkpoint_dir", ""), "test_results.json")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"\nSaved detailed test results to {out_path}")
except Exception as e:
    print(f"\nFailed to save test results: {e}")
