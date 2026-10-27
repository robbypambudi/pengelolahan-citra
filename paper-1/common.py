"""Shared paths, constants and helpers for the reproduction of

    Shetty et al., "Skin lesion classification of dermoscopic images using
    machine learning and convolutional neural network", Sci Rep 12:18134 (2022).
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT.parent / "dataverse_files"          # HAM10000 jpgs + metadata
METADATA_CSV = RAW_DIR / "HAM10000_metadata"
DATA_DIR = ROOT / "data"                           # ordered / augmented subsets
OUTPUT_DIR = ROOT / "outputs"                      # features, models, reports

# Class order used in the paper's Table 7 (0-akiec ... 6-vasc).
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

IMAGES_PER_CLASS = 100   # Step 1: 100 * 7 images (df has only 115)
SEED = 9                 # random_state used throughout Table 1


def manifest_path(subset: str) -> Path:
    """subset is 'original' (100*7) or 'augmented' (200*7)."""
    return DATA_DIR / f"{subset}_manifest.csv"


def load_manifest(subset: str) -> pd.DataFrame:
    path = manifest_path(subset)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python prepare_dataset.py` first")
    return pd.read_csv(path)


def read_image(path, size) -> np.ndarray:
    """Read a BGR image with OpenCV and resize it to (size, size)."""
    image = cv2.imread(str(path))
    if image is None:
        raise IOError(f"could not read {path}")
    return cv2.resize(image, (size, size))
