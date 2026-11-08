import numpy as np

from citra.fitur import fitur_global, haralick, histogram_hsv, hu_moments


def test_dimensi_fitur_global():
    img = np.random.default_rng(0).integers(0, 256, (220, 220, 3), dtype=np.uint8)
    assert histogram_hsv(img).shape == (512,)
    assert haralick(img).shape == (13,)
    assert hu_moments(img).shape == (7,)
    assert fitur_global(img).shape == (532,)


def test_hu_invarian_flip():
    # Hu moments invarian terhadap pencerminan (kecuali tanda momen ke-7)
    img = np.zeros((64, 64, 3), np.uint8)
    img[10:40, 5:30] = 255
    a, b = hu_moments(img), hu_moments(img[:, ::-1].copy())
    np.testing.assert_allclose(a[:6], b[:6], rtol=1e-6)
