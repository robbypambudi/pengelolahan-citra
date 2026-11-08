"""Metrik kualitas citra dan klasifikasi."""

import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def mse(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Ukuran berbeda: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def psnr(a, b, maks=255.0):
    """PSNR dalam dB; inf bila kedua citra identik."""
    e = mse(a, b)
    return float("inf") if e == 0 else float(10 * np.log10(maks**2 / e))


def ssim(a, b, maks=255.0):
    a, b = np.asarray(a), np.asarray(b)
    channel_axis = -1 if a.ndim == 3 else None
    return float(structural_similarity(a, b, data_range=maks, channel_axis=channel_axis))


def metrik_klasifikasi(y_true, y_pred, average="macro"):
    """Akurasi, presisi, recall, F1 (macro default — adil untuk kelas tak seimbang)."""
    return {
        "akurasi": float(accuracy_score(y_true, y_pred)),
        "presisi": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
