import json
import os
import random
from typing import Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm")

GATED_MODEL_HINT = (
    "google/medsiglip-448 is a gated model: accept the Health AI Developer "
    "Foundations terms on its Hugging Face page, then run `hf auth login`."
)


# =========================================
# Helpers
# =========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


# =========================================
# Annotation lists
# =========================================
def load_train_list(json_path: str, data_dir: str) -> List[Tuple[str, float]]:
    """Read {filename: MOS} from json_path and keep entries whose image exists.

    The order follows the JSON file, so a seeded shuffle reproduces the same
    cross-validation folds as the original runs.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    items, missing = [], 0
    for name, mos in data.items():
        mos = safe_float(mos)
        if np.isnan(mos):
            continue
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            items.append((path, mos))
        else:
            missing += 1
    print(f"Total images: {len(items)} | Missing: {missing}")
    return items


def index_files(root: str, exts=IMAGE_EXTS) -> Dict[str, str]:
    idx = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            lowb = fn.lower()
            stem, ext = os.path.splitext(lowb)
            if exts and ext not in exts:
                continue
            full = os.path.join(dp, fn)
            idx[lowb] = full
            idx[stem] = full
    return idx


def load_test_list(json_path: str, data_dir: str) -> List[Tuple[str, float]]:
    """Like load_train_list, but matches file names case- and extension-insensitively."""
    with open(json_path, "r") as f:
        data = json.load(f)
    file_index = index_files(data_dir)
    items, missing = [], []
    for name, mos in data.items():
        mos = safe_float(mos)
        if np.isnan(mos):
            continue
        name = str(name)
        full = os.path.join(data_dir, name)
        if os.path.exists(full):
            items.append((full, mos))
            continue
        candidates = [
            name.lower(),
            os.path.basename(name).lower(),
            os.path.splitext(name.lower())[0],
            os.path.splitext(os.path.basename(name).lower())[0],
        ]
        found = None
        for c in candidates:
            if c in file_index:
                found = file_index[c]
                break
            for ext in IMAGE_EXTS:
                if (c + ext) in file_index:
                    found = file_index[c + ext]
                    break
            if found:
                break
        if found:
            items.append((found, mos))
        else:
            missing.append(name)
    print(f"Test images: {len(items)} | Missing: {len(missing)}")
    if missing:
        print("Sample missing files:", missing[:5])
    return items


# =========================================
# Image IO
# =========================================
def read_image(path: str) -> np.ndarray:
    if os.path.splitext(path)[1].lower() == ".dcm":
        import pydicom

        img = pydicom.dcmread(path).pixel_array.astype(np.float32)
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        return (img * 255).astype(np.uint8)
    return iio.imread(path)


def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a 2D/3D image to uint8 RGB.

    Images with max <= 1 are treated as [0, 1] and scaled by 255; everything is
    then clipped to [0, 255] (instead of wrapping around on the uint8 cast).
    """
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]
    if img.max() <= 1.0:
        img = np.clip(img, 0.0, 1.0) * 255
    return np.clip(img, 0, 255).astype(np.uint8)


# =========================================
# Transforms / processor
# =========================================
def build_transforms(image_size: int):
    train_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10, fill=0),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    )
    eval_transform = T.Compose([T.Resize((image_size, image_size))])
    return train_transform, eval_transform


def build_processor(model_name: str):
    try:
        proc = AutoProcessor.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Could not load processor '{model_name}'. {GATED_MODEL_HINT}"
        ) from e
    print(f"Processor loaded: {model_name}")
    return proc


def build_text_inputs(processor, prompt: Optional[str]):
    if not prompt:
        return None
    ti = processor(text=prompt, return_tensors="pt")
    return {k: v for k, v in ti.items() if isinstance(v, torch.Tensor)}


# =========================================
# Dataset
# =========================================
class IQADataset(Dataset):
    def __init__(self, items, processor, image_size=448, transform=None, return_path=False):
        self.items = items
        self.processor = processor
        self.image_size = image_size
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, score = self.items[idx]
        try:
            img = to_uint8_rgb(read_image(path))
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {path}") from e

        pil = Image.fromarray(img)
        if self.transform is not None:
            pil = self.transform(pil)
        else:
            pil = pil.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        # Images are already resized, so the processor only rescales/normalizes.
        try:
            inputs = self.processor(
                images=pil, return_tensors="pt", do_resize=False, do_center_crop=False
            )
        except TypeError:
            inputs = self.processor(images=pil, return_tensors="pt")

        x = {"pixel_values": inputs["pixel_values"].squeeze(0)}
        y = torch.tensor(float(score), dtype=torch.float32)
        return (x, y, path) if self.return_path else (x, y)
