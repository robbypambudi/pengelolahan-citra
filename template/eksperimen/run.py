"""Kerangka eksperimen: muat data -> beri derau -> terapkan metode -> ukur -> simpan.

Contoh bawaan: perbandingan filter denoising (mean/median/gaussian) terhadap baseline.
Ganti fungsi `terapkan` dan metrik sesuai eksperimen; struktur penyimpanan hasil tetap.

Jalankan dari root proyek:  uv run python <folder-eksperimen>/run.py
"""

from pathlib import Path

import cv2
import numpy as np
import yaml
from skimage import data as skdata

from citra import baca, folder_run, ke_abu, psnr, seed_semua, simpan, simpan_metrik, ssim, tampil_grid

FOLDER = Path(__file__).resolve().parent
ROOT = next(p for p in FOLDER.parents if (p / "pyproject.toml").exists())


def muat_data(cfg):
    folder = cfg["data"]["folder"]
    if not folder:
        return {"camera": skdata.camera(), "coins": skdata.coins(), "moon": skdata.moon()}
    paths = sorted((ROOT / folder).glob(cfg["data"]["pola"]))[: cfg["data"]["maks_citra"]]
    if not paths:
        raise FileNotFoundError(f"Tidak ada citra di {ROOT / folder} dengan pola {cfg['data']['pola']}")
    return {p.stem: ke_abu(baca(p)) for p in paths}


def beri_derau(img, cfg, rng):
    d = cfg["derau"]
    if d["jenis"] == "gaussian":
        noisy = img.astype(np.float64) + rng.normal(0, d["sigma"], img.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if d["jenis"] == "salt_pepper":
        out = img.copy()
        m = rng.random(img.shape)
        out[m < d["rasio"] / 2] = 0
        out[m > 1 - d["rasio"] / 2] = 255
        return out
    raise ValueError(f"Jenis derau tidak dikenal: {d['jenis']}")


def terapkan(img, metode):
    nama, k = metode["nama"], metode.get("ukuran", 3)
    if nama == "tanpa_filter":
        return img
    if nama == "mean":
        return cv2.blur(img, (k, k))
    if nama == "median":
        return cv2.medianBlur(img, k)
    if nama == "gaussian":
        return cv2.GaussianBlur(img, (k, k), metode.get("sigma", 0))
    raise ValueError(f"Metode tidak dikenal: {nama}")


def label(metode):
    extra = "-".join(f"{k}{v}" for k, v in metode.items() if k != "nama")
    return metode["nama"] + (f"-{extra}" if extra else "")


def main():
    cfg_path = FOLDER / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    seed_semua(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    run = folder_run(FOLDER, cfg_path)

    semua_metode = [cfg["metode"]["baseline"], *cfg["metode"]["kandidat"]]
    hasil = {label(m): {"psnr": [], "ssim": []} for m in semua_metode}

    for nama, asli in muat_data(cfg).items():
        noisy = beri_derau(asli, cfg, rng)
        panel, judul = [asli], ["asli"]
        for m in semua_metode:
            out = terapkan(noisy, m)
            hasil[label(m)]["psnr"].append(psnr(asli, out))
            hasil[label(m)]["ssim"].append(ssim(asli, out))
            panel.append(out)
            judul.append(f"{label(m)}\n{hasil[label(m)]['psnr'][-1]:.2f} dB")
        simpan(run / "contoh" / f"{nama}_noisy.png", noisy)
        tampil_grid(panel, judul, kolom=len(panel), simpan_ke=run / "contoh" / f"{nama}_banding.png")

    ringkas = {
        k: {"psnr_rata": float(np.mean(v["psnr"])), "ssim_rata": float(np.mean(v["ssim"])), **v}
        for k, v in hasil.items()
    }
    simpan_metrik(run, {"nama": cfg["nama"], "seed": cfg["seed"], "hasil": ringkas})

    print(f"Hasil disimpan di {run}")
    print(f"{'metode':<28}{'PSNR (dB)':>10}{'SSIM':>8}")
    for k, v in ringkas.items():
        print(f"{k:<28}{v['psnr_rata']:>10.2f}{v['ssim_rata']:>8.4f}")


if __name__ == "__main__":
    main()
