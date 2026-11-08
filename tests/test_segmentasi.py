import cv2
import numpy as np

from citra.segmentasi import iou, mask_otsu


def _citra_sintetis(rambut=False, sudut_gelap=False):
    img = np.full((224, 224, 3), (205, 160, 145), np.uint8)   # kulit
    benar = np.zeros((224, 224), np.uint8)
    cv2.ellipse(benar, (115, 105), (55, 40), 20, 0, 360, 1, -1)
    img[benar == 1] = (95, 55, 45)                              # lesi gelap
    if rambut:
        for y0 in (40, 120, 180):
            cv2.line(img, (0, y0), (223, y0 + 30), (30, 25, 20), 2)
    if sudut_gelap:
        yy, xx = np.mgrid[:224, :224]
        sudut = np.hypot(yy - 112, xx - 112) > 150
        img[sudut] = (10, 10, 10)
    return img, benar


def test_otsu_menemukan_lesi():
    img, benar = _citra_sintetis()
    mask, info = mask_otsu(img)
    assert not info["gagal"]
    assert iou(mask, benar) > 0.9


def test_otsu_tahan_rambut_dan_sudut_gelap():
    img, benar = _citra_sintetis(rambut=True, sudut_gelap=True)
    mask, info = mask_otsu(img)
    assert not info["gagal"]
    assert iou(mask, benar) > 0.85


def test_otsu_fallback_citra_seragam():
    img = np.full((224, 224, 3), 128, np.uint8)
    mask, info = mask_otsu(img)
    assert info["gagal"]
    assert mask.shape == (224, 224) and mask.sum() > 0
