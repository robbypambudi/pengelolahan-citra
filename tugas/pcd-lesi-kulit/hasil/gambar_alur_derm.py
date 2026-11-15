"""Satu gambar PNG: alur arsitektur Derm-MobileNetV3 (A4 = M1 + M3) dengan citra asli di setiap tahap.

Sumber setiap panel:
- Citra & mask Otsu   : cache data/olahan/ham10000-256 (ISIC_0027420, sama dengan gambar Otsu), diubah ke 224 px
                        seperti saat evaluasi (interpolasi area).
- M1b, M1a            : modul asli citra/derm.py dengan parameter HASIL LATIH A4 pada fold di mana citra ini
                        menjadi data uji (eksperimen 01, run 20261108-222958).
- Peta fitur, GAP, M3 : backbone dengan bobot pralatih ImageNet (bobot hasil latih tidak disimpan) → ilustrasi.
- Probabilitas akhir  : prediksi NYATA A4 untuk citra ini (per_run/A4_r0_f*_preds.csv).

Pakai:  uv run python tugas/pcd-lesi-kulit/hasil/gambar_alur_derm.py
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
import torch.nn.functional as F  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

AKAR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AKAR))
from citra.derm import DermMobileNetV3, _softplus_inv, mask_cincin  # noqa: E402

CACHE = AKAR / "data/olahan/ham10000-256"
RUN01 = AKAR / "tugas/pcd-lesi-kulit/eksperimen/01-derm-mobilenetv3/hasil/20261108-222958"
KELUAR = Path(__file__).with_name("alur-derm-mobilenetv3.png")
GAMBAR = "ISIC_0027420"
KELAS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HIJAU, ABU, TEKS, MERAH = "#0F766E", "#6B7280", "#1F2937", "#B91C1C"


def angka(x, n):
    return f"{x:.{n}f}".replace(".", ",").replace("-", "−")


# ---------- data ----------
meta = pd.read_csv(CACHE / "meta.csv")
i = int(meta.index[meta.image == GAMBAR][0])
x = torch.from_numpy(np.asarray(np.load(CACHE / "citra.npy", mmap_mode="r")[i])).permute(2, 0, 1)[None].float() / 255
m = torch.from_numpy(np.asarray(np.load(CACHE / "mask_otsu.npy", mmap_mode="r")[i]))[None, None].float()
x = F.interpolate(x, size=224, mode="area")
m = (F.interpolate(m, size=224, mode="area") > 0.5).float()

pred = None
for f in range(5):
    df = pd.read_csv(RUN01 / f"per_run/A4_r0_f{f}_preds.csv")
    if (df.image == GAMBAR).any():
        pred, fold = df[df.image == GAMBAR].iloc[0], f
        break
pt = json.loads((RUN01 / f"per_run/A4_r0_f{fold}.json").read_text())["parameter_terpelajari"]
prob = np.array([pred[f"p_{k}"] for k in KELAS], dtype=float)

# ---------- model: struktur A4, M1 = parameter hasil latih, backbone = ImageNet ----------
model = DermMobileNetV3(m1_cc=True, m1_rambut=True, m3=True, lapisan_m3=16, lebar_cincin_px=16).eval()
with torch.no_grad():
    model.rambut.log_a.fill_(float(np.log(pt["m1_rambut_a"])))
    model.rambut.t.fill_(pt["m1_rambut_t"])
    model.rambut.gerbang.fill_(pt["m1_rambut_gerbang"])
    model.cc.theta.fill_(_softplus_inv(pt["m1_cc_p"] - 1.0))
    model.cc.gerbang.fill_(pt["m1_cc_gerbang"])
    x_rambut = model.rambut(x)
    x_cc = model.cc(x_rambut)
    h = (x_cc - model.mean) / model.std
    peta = {}
    for j, blok in enumerate(model.features):
        h = blok(h)
        peta[j] = h
    gap = model.avgpool(h).flatten(1)[0].numpy()
    w_lesi, w_cincin = model.m3.bobot(m, h.shape[-2:])
    vec_m3 = model.m3(h, m)[0].numpy()
cincin = mask_cincin(m, 16)[0, 0].numpy()


def citra_np(t):
    return t[0].permute(1, 2, 0).clamp(0, 1).numpy()


# ---------- kanvas ----------
fig = plt.figure(figsize=(24, 10.5))


def panel(x0, y0, w, h, judul=None, warna=TEKS, kecil=False):
    ax = fig.add_axes([x0, y0, w, h])
    ax.set_xticks([]); ax.set_yticks([])
    if judul:
        ax.set_title(judul, fontsize=11 if kecil else 12.5, fontweight="bold", color=warna, pad=6)
    return ax


def bingkai(ax, warna):
    for s in ax.spines.values():
        s.set_edgecolor(warna); s.set_linewidth(2.5)


def panah(p1, p2, warna=ABU, gaya="-|>", rad=0.0):
    fig.add_artist(FancyArrowPatch(p1, p2, transform=fig.transFigure, arrowstyle=gaya, mutation_scale=18,
                                   color=warna, lw=2, connectionstyle=f"arc3,rad={rad}"))


def kotak_teks(x0, y0, w, h, teks, warna_isi="#F3F4F6", warna_teks=TEKS, ukuran=12, tepi=None):
    fig.add_artist(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.004,rounding_size=0.01", transform=fig.transFigure,
                                  facecolor=warna_isi, edgecolor=tepi or warna_isi, lw=2))
    fig.text(x0 + w / 2, y0 + h / 2, teks, ha="center", va="center", fontsize=ukuran, color=warna_teks, fontweight="bold")


YT, YB, H = 0.555, 0.115, 0.30          # baris atas (jalur utama) dan bawah (jalur mask → M3)
W = 0.095
X = [0.012, 0.127, 0.242, 0.357]

# baris atas: citra → M1b → M1a → normalisasi
ax = panel(X[0], YT, W, H, "1. Input 224 × 224"); ax.imshow(citra_np(x))
ax.set_xlabel(f"{GAMBAR} · label asli: {pred['dx']}", fontsize=10.5)
ax = panel(X[1], YT, W, H, "2. M1b modul rambut", HIJAU); ax.imshow(citra_np(x_rambut)); bingkai(ax, HIJAU)
ax.set_xlabel(f"g = {angka(pt['m1_rambut_gerbang'], 2)} (rambut dipertegas)", fontsize=10.5)
ax = panel(X[2], YT, W, H, "3. M1a Shades of Gray", HIJAU); ax.imshow(citra_np(x_cc)); bingkai(ax, HIJAU)
ax.set_xlabel(f"p = {angka(pt['m1_cc_p'], 2)}, g = {angka(pt['m1_cc_gerbang'], 2)}", fontsize=10.5)
hn = ((x_cc - model.mean) / model.std)[0].permute(1, 2, 0).numpy()
ax = panel(X[3], YT, W, H, "4. Normalisasi ImageNet"); ax.imshow((hn - hn.min()) / (hn.max() - hn.min()))
ax.set_xlabel("(x − μ) / σ per kanal", fontsize=10.5)

# backbone: 4 peta fitur
XB, WB = 0.475, 0.215
fig.text(XB + WB / 2, YT + H + 0.035, "5. Backbone MobileNetV3-Large (blok 0–16, tidak diubah)", ha="center",
         fontsize=12.5, fontweight="bold", color=TEKS)
for k, (j, nama) in enumerate([(3, "blok 3 · 24 × 56²"), (6, "blok 6 · 40 × 28²"), (12, "blok 12 · 112 × 14²"), (16, "blok 16 · 960 × 7²")]):
    a = peta[j][0].abs().mean(0).numpy()
    ax = panel(XB + (k % 2) * (WB / 2 + 0.004), YT + (1 - k // 2) * (H / 2 + 0.008), WB / 2 - 0.004, H / 2 - 0.008)
    ax.imshow(a, cmap="magma")
    ax.text(0.03, 0.04, nama, transform=ax.transAxes, fontsize=10, color="white", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.55, pad=1.5, lw=0))

# GAP ⊕ M3
XG = 0.71
ax = panel(XG, YT + 0.06, 0.06, H - 0.12, "6. GAP 960")
ax.imshow(gap[:960].reshape(48, 20), cmap="viridis", aspect="auto")
fig.text(XG + 0.036, YT + 0.025, "⊕ cabang M3", ha="left", fontsize=12, fontweight="bold", color=HIJAU)

# klasifier
kotak_teks(0.79, YT + 0.09, 0.07, 0.12, "7. FC 1280\nHardswish\nDropout 0,2\n→ FC 7", ukuran=11.5)

# output nyata
ax = fig.add_axes([0.905, YT + 0.02, 0.08, H - 0.02])
warna = [MERAH if k == pred["dx"] else ABU for k in KELAS]
ax.barh(KELAS[::-1], prob[::-1], color=warna[::-1])
for yy, v in enumerate(prob[::-1]):
    ax.text(min(v, 0.75) + 0.02, yy, angka(v, 2), va="center", fontsize=10)
ax.set_xlim(0, 1); ax.tick_params(labelsize=10.5); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0,5", "1"])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("8. Output\n(prediksi nyata A4)", fontsize=12.5, fontweight="bold", pad=6)
ax.set_xlabel(f"prediksi: {KELAS[int(prob.argmax())]}\nmerah = label asli", fontsize=10.5)

# baris bawah: mask Otsu → lesi + cincin → bobot 7×7 → pooling M3 → cabang bergerbang
ax = panel(X[0], YB, W, H, "a. Mask lesi (Otsu)", HIJAU); ax.imshow(m[0, 0].numpy(), cmap="gray"); bingkai(ax, HIJAU)
ax = panel(X[1], YB, W, H, "b. Lesi + cincin 16 px", HIJAU); bingkai(ax, HIJAU)
ax.imshow(citra_np(x))
lap = np.zeros((224, 224, 4)); lap[m[0, 0].numpy() > 0] = (0.06, 0.46, 0.43, 0.45); lap[cincin > 0] = (0.96, 0.62, 0.04, 0.5)
ax.imshow(lap)
ax = panel(X[2], YB, W, H, "c. Bobot lesi 7 × 7", HIJAU); ax.imshow(w_lesi[0, 0].numpy(), cmap="Greens", vmin=0, vmax=1); bingkai(ax, HIJAU)
ax.set_xlabel("fraksi lesi per sel fitur", fontsize=10.5)
ax = panel(X[3], YB, W, H, "d. Bobot cincin 7 × 7", HIJAU); ax.imshow(w_cincin[0, 0].numpy(), cmap="Oranges", vmin=0, vmax=1); bingkai(ax, HIJAU)
ax.set_xlabel("fraksi cincin per sel fitur", fontsize=10.5)
ax = panel(XB + 0.035, YB + 0.05, WB - 0.035, H - 0.10, "e. M3 pooling peta blok 16:\n[lesi, cincin, lesi − cincin] = 3 × 960", HIJAU, kecil=True)
ax.imshow(vec_m3.reshape(3, 960), cmap="viridis", aspect="auto"); bingkai(ax, HIJAU)
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["lesi", "cincin", "selisih"], fontsize=10.5)
kotak_teks(XG - 0.005, YB + 0.10, 0.075, 0.10, "f. LN → 128\n→ 960\n× gerbang", warna_isi=HIJAU, warna_teks="white", ukuran=11.5)

# panah jalur utama
ya = YT + H / 2
for a, b in [(X[0] + W, X[1]), (X[1] + W, X[2]), (X[2] + W, X[3]), (X[3] + W, XB)]:
    panah((a + 0.002, ya), (b - 0.002, ya))
panah((XB + WB + 0.002, ya), (XG - 0.002, ya))
panah((XG + 0.062, ya), (0.79 - 0.002, ya))
panah((0.86 + 0.002, ya), (0.905 - 0.03, ya))
# panah jalur mask
panah((X[0] + W / 2, YT - 0.035), (X[0] + W / 2, YB + H + 0.04), HIJAU)
yb = YB + H / 2
for a, b in [(X[0] + W, X[1]), (X[1] + W, X[2]), (X[3] + W, XB)]:
    panah((a + 0.002, yb), (b - 0.002, yb), HIJAU)
panah((XB + WB * 0.75, YT - 0.01), (XB + WB * 0.75, YB + H - 0.05 + 0.035), HIJAU)       # peta blok 16 → M3
fig.text(XB + WB * 0.75 + 0.005, (YT + YB + H) / 2 + 0.01, "peta blok 16", fontsize=10.5, color=HIJAU)
panah((XB + WB + 0.002, yb), (XG - 0.007, yb), HIJAU)
panah((XG + 0.03, YB + 0.20 + 0.003), (XG + 0.03, YT + 0.045), HIJAU)                     # cabang → ⊕

fig.suptitle("Alur arsitektur Derm-MobileNetV3 (A4 = M1 + M3) pada satu citra HAM10000",
             fontsize=17, fontweight="bold", y=0.985)
fig.text(0.5, 0.012,
         f"Hijau = modul baru usulan. M1 memakai parameter hasil latih A4 (eksperimen 01, fold {fold}, citra ini ada di fold uji). "
         "Peta fitur, GAP, dan vektor M3 dihitung dengan bobot pralatih ImageNet (bobot hasil latih tidak disimpan), jadi hanya ilustrasi. "
         "Probabilitas pada panel 8 adalah prediksi nyata model A4.",
         ha="center", fontsize=10.5, color="#374151", wrap=True)
fig.savefig(KELUAR, dpi=120)
print(KELUAR, "fold", fold, "pred", KELAS[int(prob.argmax())], prob.round(3))
