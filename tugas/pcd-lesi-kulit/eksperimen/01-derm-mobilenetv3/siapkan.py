"""Cache data untuk eksperimen 01: citra HAM10000 di-resize ke S×S + mask Otsu.

Data mentah tidak disentuh; hasil ditulis ke data/olahan/ham10000-<S>/:
  citra.npy      (N, S, S, 3) uint8 RGB
  mask_otsu.npy  (N, S, S)    uint8 {0,1}
  meta.csv       image, dx, lesion_id, otsu_gagal, otsu_frac_area
  info.json      parameter pembuatan cache

Resize langsung 600×450 → S×S (rasio aspek tidak dipertahankan; asumsi A2 di README).

Jalankan dari root proyek (sekali, ±1–3 menit):
  uv run python tugas/pcd-lesi-kulit/eksperimen/01-derm-mobilenetv3/siapkan.py
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from citra.data import muat_ham10000
from citra.segmentasi import mask_otsu

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
PARAM_OTSU = {"k_rambut": 11, "k_morf": 5, "area_min": 0.01, "area_maks": 0.9, "maks_tepi": 0.25, "frac_fov": 0.95}


def folder_cache(ukuran):
    return ROOT / "data" / "olahan" / f"ham10000-{ukuran}"


def _olah(arg):
    path, ukuran = arg
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (ukuran, ukuran), interpolation=cv2.INTER_AREA)
    mask, info = mask_otsu(img, **PARAM_OTSU)
    return img, mask, info


def bangun_cache(ukuran=256, proses=16):
    out = folder_cache(ukuran)
    out.mkdir(parents=True, exist_ok=True)
    df = muat_ham10000()
    t0 = time.time()
    X = np.zeros((len(df), ukuran, ukuran, 3), np.uint8)
    M = np.zeros((len(df), ukuran, ukuran), np.uint8)
    gagal, frac = np.zeros(len(df), bool), np.zeros(len(df))
    with Pool(proses) as pool:
        for i, (img, mask, info) in enumerate(pool.imap(_olah, [(p, ukuran) for p in df["path"]], chunksize=32)):
            X[i], M[i], gagal[i], frac[i] = img, mask, info["gagal"], info["frac_area"]
    np.save(out / "citra.npy", X)
    np.save(out / "mask_otsu.npy", M)
    meta = df[["image", "dx", "lesion_id"]].assign(otsu_gagal=gagal, otsu_frac_area=frac)
    meta.to_csv(out / "meta.csv", index=False)
    info = {"ukuran": ukuran, "n": len(df), "param_otsu": PARAM_OTSU, "otsu_gagal": int(gagal.sum()),
            "detik": round(time.time() - t0, 1), "sumber": "data/mentah/isic2018 (ISIC 2018 Task 3 Training)"}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    return out, info


def muat_cache(ukuran=256):
    """(meta DataFrame, citra memmap, mask memmap). Bangun dulu bila belum ada."""
    out = folder_cache(ukuran)
    if not (out / "info.json").exists():
        print(f"Cache belum ada; membangun {out} ...")
        bangun_cache(ukuran)
    meta = pd.read_csv(out / "meta.csv")
    return meta, np.load(out / "citra.npy", mmap_mode="r"), np.load(out / "mask_otsu.npy", mmap_mode="r")


if __name__ == "__main__":
    ukuran = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    path, info = bangun_cache(ukuran)
    print(f"Cache: {path}")
    print(json.dumps(info, indent=2))
