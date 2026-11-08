"""Segmentasi lesi klasik: Otsu + morfologi, dengan fallback bila gagal.

Dipakai sebagai sumber *mask* untuk M3 (pooling lesi–kulit sekitar) di Derm-MobileNetV3
dan untuk RQ4. Semua fungsi bekerja pada citra RGB uint8 (H, W, 3).
"""

import cv2
import numpy as np


def _mask_fallback(h, w, frac_sumbu=0.35):
    """Elips di tengah citra; dipakai bila Otsu gagal menemukan lesi yang masuk akal."""
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (int(w * frac_sumbu), int(h * frac_sumbu)), 0, 0, 360, 1, -1)
    return m


def _isi_lubang(m):
    """Isi lubang di dalam mask biner.

    Flood fill latar dari bingkai nol 1 px di luar citra, sehingga tetap benar
    walau piksel sudut termasuk mask (mis. sudut gelap).
    """
    h, w = m.shape
    isi = np.pad(m, 1).astype(np.uint8) * 255
    cv2.floodFill(isi, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    lubang = isi[1:-1, 1:-1] == 0
    return (m.astype(bool) | lubang).astype(np.uint8)


def mask_otsu(img, k_rambut=11, k_morf=5, area_min=0.01, area_maks=0.9, maks_tepi=0.25, frac_fov=0.95):
    """Mask lesi dari citra dermoskopi RGB uint8.

    Langkah:
      1. Grayscale, lalu *closing* morfologi k_rambut×k_rambut untuk meredam rambut gelap yang tipis.
      2. Blur Gaussian, lalu ambang Otsu dari piksel di lingkaran tengah (diameter frac_fov × sisi
         terpendek); piksel ≤ ambang = lesi (lesi lebih gelap dari kulit).
      3. *Opening* + *closing* k_morf, lalu isi lubang.
      4. Pilih komponen terhubung terbesar yang tidak didominasi tepi citra
         (membuang sudut gelap/vignetting).
      5. Bila hasil tidak masuk akal (luas < area_min atau > area_maks), pakai elips tengah.

    Mengembalikan (mask uint8 {0,1}, info dict) dengan info["gagal"] = True bila fallback dipakai.
    """
    h, w = img.shape[:2]
    abu = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if k_rambut:
        abu = cv2.morphologyEx(abu, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_rambut, k_rambut)))
    abu = cv2.GaussianBlur(abu, (5, 5), 0)
    # Ambang Otsu dihitung hanya dari lingkaran tengah, agar sudut gelap (vignetting) tidak menggeser ambang.
    yy, xx = np.mgrid[:h, :w]
    tengah = np.hypot(yy - h / 2, xx - w / 2) <= frac_fov * min(h, w) / 2
    ambang, _ = cv2.threshold(abu[tengah].reshape(-1, 1), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    biner = (abu <= ambang).astype(np.uint8)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morf, k_morf))
    biner = cv2.morphologyEx(biner, cv2.MORPH_OPEN, se)
    biner = cv2.morphologyEx(biner, cv2.MORPH_CLOSE, se)
    biner = _isi_lubang(biner)

    n, label, stats, _ = cv2.connectedComponentsWithStats(biner, connectivity=8)
    tepi = np.zeros((h, w), bool)
    tepi[0, :] = tepi[-1, :] = tepi[:, 0] = tepi[:, -1] = True
    keliling_tepi = 2 * (h + w)

    terbaik, area_terbaik = 0, 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        sentuh = np.count_nonzero(tepi & (label == i)) / keliling_tepi
        if sentuh > maks_tepi:
            continue
        if area > area_terbaik:
            terbaik, area_terbaik = i, area

    frac = area_terbaik / (h * w)
    info = {"ambang_otsu": float(ambang), "frac_area": float(frac), "gagal": False}
    if terbaik == 0 or not (area_min <= frac <= area_maks):
        info["gagal"] = True
        return _mask_fallback(h, w), info
    return (label == terbaik).astype(np.uint8), info


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    gabung = np.count_nonzero(a | b)
    return 1.0 if gabung == 0 else np.count_nonzero(a & b) / gabung
