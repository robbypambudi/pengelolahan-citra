import json

import numpy as np
import pytest

from citra import baca, folder_run, ke_abu, metrik_klasifikasi, mse, psnr, simpan, simpan_metrik, ssim


def test_psnr_identik_tak_hingga():
    a = np.full((8, 8), 100, dtype=np.uint8)
    assert psnr(a, a) == float("inf")
    assert ssim(a, a) == pytest.approx(1.0)


def test_psnr_nilai_dikenal():
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.full((4, 4), 10, dtype=np.uint8)
    assert mse(a, b) == 100.0
    assert psnr(a, b) == pytest.approx(10 * np.log10(255**2 / 100))


def test_simpan_baca_bolak_balik(tmp_path):
    img = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    simpan(tmp_path / "x.png", img)
    np.testing.assert_array_equal(baca(tmp_path / "x.png"), img)  # PNG lossless, urutan RGB terjaga
    assert ke_abu(img).shape == (16, 16)


def test_metrik_klasifikasi():
    m = metrik_klasifikasi([0, 1, 1, 0], [0, 1, 0, 0])
    assert m["akurasi"] == 0.75
    assert m["confusion_matrix"] == [[2, 0], [1, 1]]


def test_folder_run(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("seed: 1\n")
    run = folder_run(tmp_path, cfg)
    simpan_metrik(run, {"psnr": 30.5})
    assert (run / "config.yaml").exists() and (run / "contoh").is_dir()
    assert json.loads((run / "metrics.json").read_text()) == {"psnr": 30.5}
