"""Satu gambar PNG: modul rambut M1b, rancangan vs hasil latih (citra/derm.py: PelemahRambut).

  abu = rata-rata kanal;  bh = closing_k(abu) − abu  (black-hat: struktur gelap tipis)
  m   = sigmoid(a · (bh − t))                       (soft-mask rambut)
  x'  = (1 − g·m) · x + g·m · closing_k(x)

Baris  = 3 citra dengan soft-mask rambut terluas dari sampel acak 2.000 citra (seed 0) + 1 citra hampir tanpa rambut.
Kolom  = asli · closing per kanal · black-hat · soft-mask · hasil desain (g = 1, a/t awal)
         · hasil terlatih (parameter A4 fold 0, eksperimen 01) · selisih terlatih − asli (×5).

Pakai:  uv run python tugas/pcd-lesi-kulit/hasil/gambar_rambut.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402


def angka(x, n):
    return f"{x:.{n}f}".replace(".", ",").replace("-", "−")


AKAR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AKAR))
from citra.derm import PelemahRambut, closing  # noqa: E402

CACHE = AKAR / "data/olahan/ham10000-256"
RUN01 = AKAR / "tugas/pcd-lesi-kulit/eksperimen/01-derm-mobilenetv3/hasil/20261108-222958"
KELUAR = Path(__file__).with_name("modul-rambut-m1b.png")
K = 11

meta = pd.read_csv(CACHE / "meta.csv")
citra = np.load(CACHE / "citra.npy", mmap_mode="r")
pt = json.loads((RUN01 / "per_run/A4_r0_f0.json").read_text())["parameter_terpelajari"]


def modul(a, t, g):
    m = PelemahRambut(k=K)
    with torch.no_grad():
        m.log_a.fill_(float(np.log(a))); m.t.fill_(t); m.gerbang.fill_(g)
    return m.eval()


desain = modul(100.0, 0.08, 1.0)          # nilai awal a, t; gerbang penuh (= operator tetap)
terlatih = modul(pt["m1_rambut_a"], pt["m1_rambut_t"], pt["m1_rambut_gerbang"])


def ke_tensor(i):
    return torch.from_numpy(np.asarray(citra[i])).permute(2, 0, 1)[None].float() / 255


# pilih contoh: luas soft-mask rambut (desain) pada sampel acak
rng = np.random.default_rng(0)
sampel = rng.choice(len(meta), 2000, replace=False)
with torch.no_grad():
    luas = np.array([float((desain.soft_mask(ke_tensor(i)) > 0.5).float().mean()) for i in sampel])
urut = sampel[np.argsort(luas)[::-1]]
contoh = list(urut[:3]) + [int(sampel[np.argsort(luas)[len(luas) // 2]])]
label = ["rambut lebat"] * 3 + ["rambut sedikit (median)"]

KOLOM = ["1. Citra asli", f"2. Closing {K}×{K} per kanal\n(isian pengganti rambut)", "3. Black-hat\nclosing(abu) − abu",
         "4. Soft-mask rambut m\nsigmoid(a·(bh − t))", "5. Bila dipaksa g = 1\n(sesuai rancangan)",
         f"6. Hasil nyata setelah dilatih\ng = {angka(pt['m1_rambut_gerbang'], 2)} → rambut dipertegas", "7. Selisih terlatih − asli\n(diperkuat ×5)"]
fig, axs = plt.subplots(len(contoh), len(KOLOM), figsize=(21, 3.2 * len(contoh) + 1.0))
with torch.no_grad():
    for r, i in enumerate(contoh):
        x = ke_tensor(i)
        abu = x.mean(1, keepdim=True)
        bh = (closing(abu, K) - abu)[0, 0].numpy()
        sm = desain.soft_mask(x)[0, 0].numpy()
        tampil = [x, closing(x, K), None, None, desain(x), terlatih(x), None]
        for c in range(len(KOLOM)):
            ax = axs[r, c]
            if c == 2:
                ax.imshow(bh, cmap="magma", vmin=0, vmax=max(0.15, float(bh.max())))
            elif c == 3:
                ax.imshow(sm, cmap="gray", vmin=0, vmax=1)
                ax.set_xlabel(f"{(sm > 0.5).mean()*100:.1f}% piksel m > 0,5".replace(".", ","), fontsize=11)
            elif c == 6:
                d = (terlatih(x) - x)[0].permute(1, 2, 0).numpy()
                ax.imshow(np.clip(0.5 + 5 * d, 0, 1))
            else:
                ax.imshow(tampil[c][0].permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(KOLOM[c], fontsize=12, fontweight="bold", pad=8)
        axs[r, 0].set_ylabel(f"{meta.dx[i]} · {label[r]}\n{meta.image[i]}", fontsize=12)
for r in range(len(contoh)):
    for c in (4, 5):
        for s in axs[r, c].spines.values():
            s.set_edgecolor("#0F766E" if c == 4 else "#B91C1C"); s.set_linewidth(2.5)
fig.suptitle("Modul rambut M1b: dirancang untuk menghapus rambut, tetapi setelah dilatih model justru mempertegasnya (g < 0)",
             fontsize=15, fontweight="bold", y=0.995)
ket = ("Desain: a = 100, t = 0,08, g = 1 (rambut diganti warna sekitar). Terlatih (eksperimen 01, A4 fold 0): "
       f"a = {angka(pt['m1_rambut_a'], 0)}, t = {angka(pt['m1_rambut_t'], 3)}, g = {angka(pt['m1_rambut_gerbang'], 3)}"
       + ". Gerbang negatif di kelima fold A1/A4 (−0,08 s.d. −0,25): model belajar sedikit MEMPERTEGAS rambut, bukan menghapusnya.")
fig.text(0.5, 0.006, ket, ha="center", fontsize=11, color="#374151")
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
fig.savefig(KELUAR, dpi=130)
print(KELUAR, [meta.image[i] for i in contoh])
