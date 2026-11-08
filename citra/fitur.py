"""Fitur global ala Shetty dkk. (2022): histogram warna HSV, Haralick, Hu moments.

Input citra BGR uint8 (seperti cv2.imread), sesuai pipeline OpenCV yang dipakai paper.
"""

import cv2
import mahotas
import numpy as np


def hu_moments(bgr):
    """7 invarian momen Hu dari citra grayscale (bentuk)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(gray)).flatten()


def haralick(bgr):
    """13 fitur tekstur Haralick (GLCM), dirata-rata atas 4 arah."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return mahotas.features.haralick(gray).mean(axis=0)


def histogram_hsv(bgr, bins=8):
    """Histogram 3D HSV bins^3 yang dinormalisasi (warna)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def fitur_global(bgr, bins=8):
    """Gabungan [histogram | haralick | hu] = bins^3 + 13 + 7 dimensi (532 untuk bins=8)."""
    return np.hstack([histogram_hsv(bgr, bins), haralick(bgr), hu_moments(bgr)]).astype(np.float64)
