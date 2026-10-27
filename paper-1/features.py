"""Global feature descriptors (paper section "Feature extraction", Fig. 2).

    colour  -> colour histogram in HSV space (8x8x8 bins = 512 values)
    texture -> Haralick texture features on the grey image (13 values)
    shape   -> Hu moments on the grey image (7 values)

The three descriptors are concatenated into a single 532-dim global feature.
"""
import cv2
import mahotas
import numpy as np

HIST_BINS = 8


def fd_hu_moments(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(gray)).flatten()


def fd_haralick(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Mean over the four GLCM directions.
    return mahotas.features.haralick(gray).mean(axis=0)


def fd_histogram(image: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def global_feature(image: np.ndarray) -> np.ndarray:
    return np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)])
