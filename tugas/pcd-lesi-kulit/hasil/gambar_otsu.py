"""Satu gambar PNG: langkah segmentasi Otsu dan kegunaan mask-nya di eksperimen.

Baris = contoh citra (mel, nv, bkl dengan luas mask mendekati median kelasnya, + satu kasus gagal → elips).
Kolom = citra asli · abu-abu + closing (rambut diredam) · histogram + ambang Otsu · mask Otsu ·
        M3: lesi + cincin 32 px · H5d: hanya lesi · H5d: hanya latar (lesi + 16 px ditutup).

Pakai:  uv run python tugas/pcd-lesi-kulit/hasil/gambar_otsu.py
"""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AKAR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AKAR))
from citra.segmentasi import mask_otsu  # noqa: E402

CACHE = AKAR / "data/olahan/ham10000-256"
KELUAR = Path(__file__).with_name("otsu-kegunaan.png")
P = dict(k_rambut=11, k_morf=5, frac_fov=0.95)
MEAN = np.array([0.485, 0.456, 0.406]) * 255
CINCIN, DILASI_TUTUP = 32, 16

meta = pd.read_csv(CACHE / "meta.csv")
citra = np.load(CACHE / "citra.npy", mmap_mode="r")
mask = np.load(CACHE / "mask_otsu.npy", mmap_mode="r")


def contoh(dx):
    ok = meta[(meta.dx == dx) & (~meta.otsu_gagal)]
    return int((ok.otsu_frac_area - ok.otsu_frac_area.median()).abs().idxmin())


idx = [(contoh("mel"), "mel"), (contoh("nv"), "nv"), (contoh("bkl"), "bkl (kontras rendah,\nmask meleset)"),
       (int(meta.index[meta.otsu_gagal][0]), f"{meta.dx[meta.index[meta.otsu_gagal][0]]} (Otsu gagal)")]


def dilasi(m, px):
    k = 2 * px + 1
    return cv2.dilate(m, np.ones((k, k), np.uint8))


def kontur(ax, m, warna, lw=1.8):
    ax.contour(m, levels=[0.5], colors=[warna], linewidths=lw)


KOLOM = ["1. Citra asli", "2. Abu-abu + closing 11×11\n(rambut diredam)", "3. Histogram + ambang Otsu",
         "4. Mask Otsu (lesi)", f"Pakai: M3 pooling\nlesi + cincin {CINCIN} px", "Pakai: H5d hanya lesi\n(latar ditutup)",
         f"Pakai: H5d hanya latar\n(lesi + {DILASI_TUTUP} px ditutup)"]
fig, axs = plt.subplots(len(idx), len(KOLOM), figsize=(21, 3.2 * len(idx) + 0.9))
for r, (i, label) in enumerate(idx):
    img = np.asarray(citra[i])
    m = np.asarray(mask[i])
    m2, info = mask_otsu(img, **P)
    assert (m2 == m).all(), "mask cache ≠ mask_otsu: parameter berbeda"
    h, w = m.shape
    abu = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abu = cv2.morphologyEx(abu, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    abu = cv2.GaussianBlur(abu, (5, 5), 0)
    yy, xx = np.mgrid[:h, :w]
    tengah = np.hypot(yy - h / 2, xx - w / 2) <= 0.95 * min(h, w) / 2

    ax = axs[r, 0]; ax.imshow(img); ax.set_ylabel(f"{label}\n{meta.image[i]}", fontsize=12)
    ax = axs[r, 1]; ax.imshow(abu, cmap="gray", vmin=0, vmax=255)
    ax = axs[r, 2]
    ax.hist(abu[tengah].ravel(), bins=64, range=(0, 255), color="#9CA3AF")
    ax.axvline(info["ambang_otsu"], color="#B91C1C", lw=2)
    ax.text(info["ambang_otsu"] + 4, ax.get_ylim()[1] * 0.85, f"T = {info['ambang_otsu']:.0f}", color="#B91C1C", fontsize=11)
    ax.set_xlim(0, 255); ax.set_yticks([]); ax.tick_params(labelsize=9)
    ax.text(3, ax.get_ylim()[1] * 0.85, "← lesi", fontsize=10, color="#374151")
    ax = axs[r, 3]; ax.imshow(m, cmap="gray", vmin=0, vmax=1)
    ax.set_xlabel(f"luas mask {info['frac_area']*100:.0f}% citra" if not info["gagal"] else "fallback: elips tengah", fontsize=11)
    ax = axs[r, 4]; ax.imshow(img)
    cincin = (dilasi(m, CINCIN) > 0) & (m == 0)
    lapis = np.zeros((h, w, 4)); lapis[m > 0] = (0.06, 0.46, 0.43, 0.35); lapis[cincin] = (0.96, 0.62, 0.04, 0.35)
    ax.imshow(lapis); kontur(ax, m, "#0F766E"); kontur(ax, dilasi(m, CINCIN), "#D97706")
    ax = axs[r, 5]
    tampil = img.copy(); tampil[m == 0] = MEAN.astype(np.uint8); ax.imshow(tampil)
    ax = axs[r, 6]
    tampil = img.copy(); tampil[dilasi(m, DILASI_TUTUP) > 0] = MEAN.astype(np.uint8); ax.imshow(tampil)
    for c in range(len(KOLOM)):
        if c != 2:
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        if r == 0:
            axs[r, c].set_title(KOLOM[c], fontsize=12, fontweight="bold", pad=8)
for c in range(4, 7):
    for r in range(len(idx)):
        for s in axs[r, c].spines.values():
            s.set_edgecolor("#0F766E"); s.set_linewidth(2)
fig.suptitle("Segmentasi Otsu pada HAM10000 dan kegunaannya: langkah 1–4 membuat mask; kolom hijau = dipakai di eksperimen",
             fontsize=15, fontweight="bold", y=0.995)
n_gagal = int(meta.otsu_gagal.sum())
n_total = f"{len(meta):,}".replace(",", ".")
fig.text(0.5, 0.005, "Hijau = lesi, jingga = cincin kulit sekitar (M3 Derm-MobileNetV3). Area tertutup diisi rata-rata ImageNet. "
         f"Otsu gagal pada {n_gagal} dari {n_total} citra → elips tengah. Mask juga dipakai augmentasi tukar latar (eksperimen 05).",
         ha="center", fontsize=11, color="#374151")
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
fig.savefig(KELUAR, dpi=130)
print(KELUAR)
