"""Baca/simpan citra. Konvensi: array numpy RGB (bukan BGR), dtype uint8."""

from pathlib import Path

import cv2
import numpy as np


def baca(path, abu=False):
    """Baca citra sebagai RGB uint8 (atau grayscale bila abu=True)."""
    path = Path(path)
    flag = cv2.IMREAD_GRAYSCALE if abu else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Citra tidak bisa dibaca: {path}")
    return img if abu else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def simpan(path, img):
    """Simpan citra RGB/grayscale; folder induk dibuat otomatis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), img):
        raise IOError(f"Gagal menyimpan: {path}")


def ke_abu(img):
    """RGB -> grayscale (luminans ITU-R BT.601, sama seperti OpenCV)."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
