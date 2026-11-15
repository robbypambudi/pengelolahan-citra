"""Deck hasil eksperimen 05 (v2). Semua angka dibaca dari metrics.json dan per_run/ run yang diberikan.

Pakai:
  uv run --no-project --with python-pptx python tugas/pcd-lesi-kulit/hasil/buat_ppt_eks05.py \
      tugas/pcd-lesi-kulit/eksperimen/05-regularisasi/hasil/20261109-164154 \
      tugas/pcd-lesi-kulit/hasil/eksperimen-05-regularisasi.pptx
"""
import csv
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
# run eksperimen 03 (Derm-MobileNetV3 / M3, protokol dan split sama dengan eksperimen 05)
RUN03 = Path(sys.argv[3]) if len(sys.argv) > 3 else RUN.parents[2] / "03-derm-v2/hasil/20261109-054525"
d = json.loads((RUN / "metrics.json").read_text())
d3 = json.loads((RUN03 / "metrics.json").read_text())
R, B = d["ringkasan"], d["banding_a3_vs_A0"]
V = ["A0", "R-ls", "R-mixup", "R-cutmix", "R-sd", "R-10ep", "R-latar"]
NAMA = {"A0": "A0 (acuan)", "R-ls": "Label smoothing", "R-mixup": "Mixup", "R-cutmix": "CutMix",
        "R-sd": "Stochastic depth", "R-10ep": "10 epoch", "R-latar": "Tukar latar"}
KELAS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
terpilih = d["a3_terpilih_menurut_validasi"]


def m(v, k, met="bacc"):
    return R[f"{v}|{k}"][met]["rata"]


def sd(v, k, met="bacc"):
    return R[f"{v}|{k}"][met]["sd"]


def b(v, k):
    return B[f"{v} − A0 | {k}"]


def f(x, n=4):
    return f"{x:.{n}f}".replace(".", ",")


def poin(x):
    return f"{x*100:+.2f}".replace(".", ",").replace("-", "−")


def per_run(v, i):
    return json.loads((RUN / f"per_run/{v}_r0_f{i}.json").read_text())


# ---------- data turunan ----------
kurva = {}
for v in V:
    H = [per_run(v, i)["riwayat"] for i in range(5)]
    n = len(H[0])
    kurva[v] = ([sum(h[e]["loss"] for h in H) / 5 for e in range(n)],
                [sum(h[e]["bacc_val"] for h in H) / 5 for e in range(n)])


def confusion(v):
    """Gabungan 5 fold uji, keputusan argmax. cm[asli][prediksi] = jumlah citra."""
    cm = [[0] * 7 for _ in range(7)]
    for i in range(5):
        with open(RUN / f"per_run/{v}_r0_f{i}_preds.csv") as fh:
            for r in csv.DictReader(fh):
                p = [float(r[f"p_{k}"]) for k in KELAS]
                cm[KELAS.index(r["dx"])][p.index(max(p))] += 1
    return cm


CM = {v: confusion(v) for v in ("A0", terpilih)}
recall = {v: [CM[v][i][i] / sum(CM[v][i]) for i in range(7)] for v in CM}
n_kelas = [sum(CM["A0"][i]) for i in range(7)]

# ---------- gaya ----------
TEKS = RGBColor(0x1F, 0x29, 0x37)
ABU = RGBColor(0x6B, 0x72, 0x80)
ABU_MUDA = RGBColor(0xE5, 0xE7, 0xEB)
LATAR_KARTU = RGBColor(0xF3, 0xF4, 0xF6)
PUTIH = RGBColor(0xFF, 0xFF, 0xFF)
AKSEN = RGBColor(0x0F, 0x76, 0x6E)
AKSEN_2 = RGBColor(0x99, 0xC9, 0xC3)
AKSEN_TINT = RGBColor(0xDC, 0xEF, 0xEC)
JINGGA = RGBColor(0xC2, 0x41, 0x0C)
UNGU = RGBColor(0x6D, 0x28, 0xD9)
FONT = "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
KOSONG = prs.slide_layouts[6]
nomor = [0]


def kotak_teks(s, x, y, w, h, teks, ukuran=16, tebal=False, warna=TEKS, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, t in enumerate(teks if isinstance(teks, list) else [teks]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = t
        r.font.size, r.font.bold, r.font.name = Pt(ukuran), tebal, FONT
        r.font.color.rgb = warna
    return tb


def poin_poin(s, x, y, w, h, items, ukuran=16):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        r0 = p.add_run()
        r0.text = "•  "
        r0.font.size, r0.font.name = Pt(ukuran), FONT
        r0.font.color.rgb = AKSEN
        for t, tebal in (it if isinstance(it, list) else [(it, False)]):
            r = p.add_run()
            r.text = t
            r.font.size, r.font.bold, r.font.name = Pt(ukuran), tebal, FONT
            r.font.color.rgb = TEKS


def kotak(s, x, y, w, h, baris, isi=LATAR_KARTU, garis=None, ukuran=13, warna=TEKS, tebal_pertama=True):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.12
    sh.fill.solid()
    sh.fill.fore_color.rgb = isi
    if garis is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = garis
        sh.line.width = Pt(1.5)
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    for i, t in enumerate(baris):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = t
        r.font.size = Pt(ukuran if i == 0 else ukuran - 1)
        r.font.bold = tebal_pertama and i == 0
        r.font.name = FONT
        r.font.color.rgb = warna
    return sh


def panah(s, x1, y1, x2, y2, warna=ABU):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = warna
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    ujung = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(ujung)
    return c


def slide(judul):
    s = prs.slides.add_slide(KOSONG)
    nomor[0] += 1
    kotak_teks(s, 0.6, 0.45, 12.1, 0.8, judul, 30, True)
    kotak_teks(s, 0.6, 7.0, 9, 0.3, "Eksperimen 05 · Regularisasi pelatihan MobileNetV3-Large · HAM10000", 10, warna=ABU)
    kotak_teks(s, 11.7, 7.0, 1.0, 0.3, str(nomor[0]), 10, warna=ABU, align=PP_ALIGN.RIGHT)
    return s


def isi_sel(cell, teks, ukuran, tebal=False, warna=TEKS, latar=PUTIH, align=PP_ALIGN.CENTER):
    cell.text = ""
    p = cell.text_frame.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = teks
    r.font.size, r.font.name, r.font.bold = Pt(ukuran), FONT, tebal
    r.font.color.rgb = warna
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = cell.margin_right = Inches(0.05)
    cell.margin_top = cell.margin_bottom = Inches(0.02)
    cell.fill.solid()
    cell.fill.fore_color.rgb = latar


def tabel(s, x, y, w, kolom, baris, lebar=None, ukuran=13, sorot=None):
    t = s.shapes.add_table(len(baris) + 1, len(kolom), Inches(x), Inches(y), Inches(w), Inches(0.4 * (len(baris) + 1))).table
    if lebar:
        for i, lw in enumerate(lebar):
            t.columns[i].width = Inches(lw)
    for r in range(len(baris) + 1):
        t.rows[r].height = Inches(0.4)
        for c in range(len(kolom)):
            disorot = sorot is not None and r - 1 == sorot
            isi_sel(t.cell(r, c), kolom[c] if r == 0 else baris[r - 1][c], ukuran,
                    tebal=r == 0 or disorot, warna=PUTIH if r == 0 else TEKS,
                    latar=TEKS if r == 0 else AKSEN_TINT if disorot else PUTIH if r % 2 else LATAR_KARTU,
                    align=PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER)
    return t


def gaya_sumbu(ch, minimum, maksimum, fmt="0.000"):
    ch.font.size, ch.font.name = Pt(12), FONT
    ch.font.color.rgb = TEKS
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = minimum, maksimum
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = ABU_MUDA
    va.format.line.fill.background()
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = fmt, False
    ch.category_axis.format.line.color.rgb = ABU_MUDA


def label_data(ch, fmt="0.000"):
    pl = ch.plots[0]
    pl.has_data_labels = True
    dl = pl.data_labels
    dl.number_format, dl.number_format_is_linked = fmt, False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(11)
    return pl


def judul_grafik(ch, teks):
    ch.has_title = True
    ch.chart_title.text_frame.text = teks
    r = ch.chart_title.text_frame.paragraphs[0].runs[0]
    r.font.size, r.font.bold, r.font.name = Pt(14), True, FONT


# ========== 1. Judul ==========
s = prs.slides.add_slide(KOSONG)
nomor[0] += 1
kotak_teks(s, 0.8, 2.3, 11.7, 1.4, "Regularisasi Pelatihan pada MobileNetV3-Large untuk Klasifikasi Lesi Kulit", 38, True)
kotak_teks(s, 0.8, 3.9, 11.7, 0.6, "Eksperimen 05 · HAM10000 · 7 varian × 5 fold per lesi", 20, warna=AKSEN)
kotak_teks(s, 0.8, 5.6, 11.7, 0.9, ["Pengolahan Citra Digital · Magister Informatika ITS",
                                     f"Run hasil/20261109-164154 · draf, {f(d['durasi_jam'], 2)} jam GPU"], 14, warna=ABU)

# ========== 2. Ringkasan ==========
s = slide("Ringkasan: menang di argmax, belum lolos aturan utama")
kartu = [
    (poin(b("R-mixup", "argmax")["selisih_bacc_rata"]) + " poin",
     ["Mixup vs A0, keputusan argmax",
      f"unggul di {b('R-mixup','argmax')['n_fold_positif']}/5 fold · p_holm {f(b('R-mixup','argmax')['p_holm'],3)}"], AKSEN),
    (poin(b(terpilih, "bias_val")["selisih_bacc_rata"]) + " poin",
     ["CutMix (terpilih menurut validasi), +bias_val",
      f"p_holm {f(b(terpilih,'bias_val')['p_holm'],3)} · di bawah ambang 2 poin"], TEKS),
    (f"{f(m('A0','argmax','ece'))} → {f(m('R-cutmix','argmax','ece'))}",
     ["ECE A0 → CutMix", "model jauh lebih terkalibrasi"], AKSEN),
]
for i, (angka, ket, w) in enumerate(kartu):
    x = 0.6 + i * 4.1
    kt = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.6), Inches(3.8), Inches(2.6))
    kt.fill.solid(); kt.fill.fore_color.rgb = LATAR_KARTU; kt.line.fill.background(); kt.shadow.inherit = False
    kotak_teks(s, x + 0.3, 1.9, 3.2, 0.9, angka, 36, True, w)
    kotak_teks(s, x + 0.3, 2.95, 3.2, 1.1, ket, 14)
poin_poin(s, 0.6, 4.6, 12.1, 2.2, [
    [("Hipotesis \"regularisasi > A0\" tidak didukung ", True), ("menurut aturan yang ditetapkan sebelum run.", False)],
    [("Mixup, CutMix, dan stochastic depth ", True), ("mengalahkan A0 di semua 5 fold pada argmax (+2,6 s.d. +3,4 poin), tetapi belum bermakna setelah koreksi Holm.", False)],
    [("Penyetelan bias per kelas ", True), ("membantu A0, tetapi menghapus keunggulan model yang sudah diregularisasi.", False)],
], 16)

# ========== 3. Arsitektur ==========
s = slide("Arsitektur: MobileNetV3-Large (A0) dan titik kerja regularisasi")
W, G = 1.85, 0.2
baris1 = [
    (["Input", "224 × 224 × 3"], PUTIH, ABU),
    (["Stem", "conv 3×3, s2", "16 kanal · 112²"], LATAR_KARTU, None),
    (["Blok 1", "bneck 3×3", "16 kanal · 112²"], LATAR_KARTU, None),
    (["Blok 2–3", "bneck 3×3", "24 kanal · 56²"], LATAR_KARTU, None),
    (["Blok 4–6 · SE", "bneck 5×5", "40 kanal · 28²"], AKSEN_TINT, None),
    (["Blok 7–10", "bneck 3×3", "80 kanal · 14²"], LATAR_KARTU, None),
]
baris2 = [
    (["Blok 11–12 · SE", "bneck 3×3", "112 kanal · 14²"], AKSEN_TINT, None),
    (["Blok 13–15 · SE", "bneck 5×5", "160 kanal · 7²"], AKSEN_TINT, None),
    (["Conv 1×1", "960 kanal · 7²"], LATAR_KARTU, None),
    (["Global avg pool", "960"], LATAR_KARTU, None),
    (["FC 1280", "Hardswish", "Dropout 0,2"], LATAR_KARTU, None),
    (["FC 7", "softmax 7 kelas"], PUTIH, ABU),
]
Y1, Y2, H = 1.55, 3.25, 1.15
for j, (baris, y) in enumerate(((baris1, Y1), (baris2, Y2))):
    for i, (teks, isi, garis) in enumerate(baris):
        x = 0.6 + i * (W + G)
        kotak(s, x, y, W, H, teks, isi=isi, garis=garis, ukuran=13)
        if i < len(baris) - 1:
            panah(s, x + W, y + H / 2, x + W + G, y + H / 2)
# sambungan baris 1 → baris 2
x_akhir = 0.6 + 5 * (W + G) + W / 2
panah(s, x_akhir, Y1 + H, x_akhir, Y1 + H + 0.25)
garis_balik = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x_akhir), Inches(Y1 + H + 0.25), Inches(0.6 + W / 2), Inches(Y1 + H + 0.25))
garis_balik.line.color.rgb, garis_balik.line.width = ABU, Pt(1.5)
panah(s, 0.6 + W / 2, Y1 + H + 0.25, 0.6 + W / 2, Y2)
kotak_teks(s, 0.6, 4.55, 12.1, 0.35, "Bobot pralatih ImageNet (IMAGENET1K_V1); kepala akhir 1280 → 7 kelas. 4.210.999 parameter, 430,6 MFLOPs — sama untuk semua varian. SE = squeeze-and-excitation; Blok = inverted residual (bneck).", 12, warna=ABU)
# titik kerja regularisasi
kotak_teks(s, 0.6, 5.05, 12.1, 0.4, "Di mana regularisasi bekerja (hanya saat latih)", 16, True)
titik = [
    ("Input batch", "Mixup (α 0,2) · CutMix (α 1,0) · tukar latar (p 0,5), dijalankan sesudah augmentasi geometri"),
    ("10 blok residual", "Stochastic depth pada blok ber-skip (1, 3, 5, 6, 8, 9, 10, 12, 14, 15), p naik linear 0,02 → 0,20"),
    ("Loss", "Label smoothing ε 0,1 di atas logit-adjusted cross-entropy (τ = 1); 10 epoch = jadwal lebih pendek"),
]
for i, (lbl, ket) in enumerate(titik):
    y = 5.5 + i * 0.45
    kotak(s, 0.6, y, 2.1, 0.36, [lbl], isi=AKSEN, ukuran=12, warna=PUTIH)
    kotak_teks(s, 2.9, y + 0.05, 9.8, 0.36, ket, 13)

# ========== 3b. Metode usulan vs A0 ==========
s = slide("Metode usulan Derm-MobileNetV3 vs A0 (berdampingan)")
BH, BG, Y0 = 0.48, 0.15, 1.85
y_row = [Y0 + i * (BH + BG) for i in range(8)]
XA, WA = 0.6, 3.6          # kolom A0
XD, WD = 5.0, 3.6          # kolom utama Derm
XS, WS = 9.1, 3.6          # kolom samping Derm (mask + M3)
kotak_teks(s, XA, 1.35, WA, 0.4, "A0 · MobileNetV3-Large polos", 16, True)
kotak_teks(s, XD, 1.35, WD + WS + 0.5, 0.4, "Derm-MobileNetV3 (usulan) · backbone sama + modul pengolahan citra", 16, True, AKSEN)
garis_pemisah = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(4.6), Inches(1.4), Inches(4.6), Inches(6.75))
garis_pemisah.line.color.rgb, garis_pemisah.line.width = ABU_MUDA, Pt(1)

lama = dict(isi=LATAR_KARTU, ukuran=12)
baru = dict(isi=AKSEN, warna=PUTIH, ukuran=12)
# A0: baris 0, 3, 4, 5, 6, 7 (baris 1–2 kosong = tidak ada M1)
a0 = {0: ["Citra RGB 224 × 224"], 3: ["Normalisasi ImageNet"], 4: ["Backbone MobileNetV3-Large", "blok 0–16 → 960 × 7²"],
      5: ["Global average pooling → 960"], 6: ["FC 1280 · Hardswish · Dropout 0,2"], 7: ["FC 7 kelas"]}
urut_a0 = sorted(a0)
for i in urut_a0:
    kotak(s, XA, y_row[i], WA, BH, a0[i], **lama)
for p, q in zip(urut_a0, urut_a0[1:]):
    panah(s, XA + WA / 2, y_row[p] + BH, XA + WA / 2, y_row[q])
# Derm: kolom utama
derm = {0: (["Citra RGB 224 × 224"], lama),
        1: (["M1b Pelemah rambut", "black-hat + soft-mask, gerbang g"], baru),
        2: (["M1a Shades of Gray terlatih", "eksponen p dipelajari, gerbang g"], baru),
        3: (["Normalisasi ImageNet"], lama),
        4: (["Backbone MobileNetV3-Large", "blok 0–16 → 960 × 7² (tidak diubah)"], lama),
        5: (["GAP 960  ⊕  cabang M3"], dict(isi=AKSEN_TINT, garis=AKSEN, ukuran=12)),
        6: (["FC 1280 · Hardswish · Dropout 0,2"], lama),
        7: (["FC 7 kelas"], lama)}
for i, (teks, gaya) in derm.items():
    kotak(s, XD, y_row[i], WD, BH, teks, **gaya)
    if i < 7:
        panah(s, XD + WD / 2, y_row[i] + BH, XD + WD / 2, y_row[i + 1])
# Derm: kolom samping
kotak(s, XS, y_row[0], WS, BH, ["Mask lesi (Otsu)"], isi=PUTIH, garis=ABU, ukuran=12)
y_m3 = y_row[2]
h_m3 = y_row[4] + BH - y_m3
kotak(s, XS, y_m3, WS, h_m3, ["M3 Pooling lesi–sekitar", "fitur dari blok 12 (112 × 14²)",
                               "pool di lesi · cincin 32 px · selisih", "→ 336"], **baru)
kotak(s, XS, y_row[5], WS, BH, ["LN → 128 → 960, × gerbang"], **baru)
panah(s, XS + WS / 2, y_row[0] + BH, XS + WS / 2, y_m3)                    # mask → M3
panah(s, XD + WD, y_row[4] + BH / 2, XS, y_row[4] + BH / 2)                # blok 12 → M3
panah(s, XS + WS / 2, y_row[4] + BH, XS + WS / 2, y_row[5])                # M3 → cabang
panah(s, XS, y_row[5] + BH / 2, XD + WD, y_row[5] + BH / 2)                # cabang → ⊕
b03 = d3["biaya"]["A3-b12-r32"]
kotak_teks(s, XS, y_row[6], WS, 1.1, [
    "Hijau = modul baru. Semua mulai sebagai identitas (gerbang / Linear akhir = 0), jadi langkah ke-0 = A0.",
    f"M3 (b12-r32): +{b03['parameter_tambahan']:,} parameter".replace(",", ".") + f", {f(b03['flops']/1e6,1)} MFLOPs.",
    "Diuji: M3 di eksperimen 03 (protokol sama); M1 + M3 (A4) di eksperimen 01 (loss berbeda).",
], 11, warna=ABU)

# ========== 4. Alur training ==========
s = slide("Alur pelatihan dan evaluasi")
alur = [
    ["HAM10000", "10.015 citra", "7 kelas"],
    ["5 fold per lesi", "StratifiedGroupKFold", "seed 0"],
    ["Pisah fold latih", "90% lesi → latih", "10% lesi → validasi dalam"],
    ["Latih 20 epoch", "AdamW · batch 64", "warmup 1 + cosine"],
    ["Keputusan", "argmax, atau", "+bias_val per kelas"],
    ["Uji fold ke-k", "BACC, AUROC, F1, ECE", "tidak dipakai memilih"],
]
W, G, Y, H = 1.85, 0.2, 1.6, 1.4
for i, teks in enumerate(alur):
    x = 0.6 + i * (W + G)
    kotak(s, x, Y, W, H, teks, isi=AKSEN if i == 5 else LATAR_KARTU, warna=PUTIH if i == 5 else TEKS, ukuran=14)
    if i < len(alur) - 1:
        panah(s, x + W, Y + H / 2, x + W + G, Y + H / 2)
kotak_teks(s, 0.6, 3.35, 12.1, 0.35, "Diulang untuk 5 fold × 7 varian = 35 run. Split dan urutan batch identik dengan eksperimen 01–04.", 13, warna=ABU)
kol = [
    ("Pengaturan latih (semua varian)", [
        "Input 224 px dari cache 256 px; augmentasi rotasi, flip, skala 0,8–1, geser 0,1, kecerahan/kontras 0,1",
        "lr 3·10⁻⁴, weight decay 10⁻⁴, AMP; bobot epoch terakhir yang dievaluasi",
        "Loss logit-adjusted (τ = 1) untuk ketidakseimbangan kelas",
    ]),
    ("Aturan keputusan (dikunci sebelum run)", [
        "Varian terpilih = BACC validasi dalam tertinggi",
        "Menang bila Δ BACC (+bias_val) ≥ 2 poin dan p_holm < 0,05",
        "Uji t resampled terkoreksi, Holm atas 12 perbandingan",
    ]),
]
for i, (jdl, isi) in enumerate(kol):
    x = 0.6 + i * 6.25
    kotak_teks(s, x, 4.0, 5.85, 0.4, jdl, 16, True)
    poin_poin(s, x, 4.5, 5.85, 2.4, isi, 14)

# ========== 5. Grafik BACC ==========
s = slide("BACC uji: regularisasi naik di argmax, tidak di +bias_val")
cd = CategoryChartData()
cd.categories = [NAMA[v] for v in V]
cd.add_series("Argmax", [round(m(v, "argmax"), 4) for v in V])
cd.add_series("+bias_val (keputusan utama)", [round(m(v, "bias_val"), 4) for v in V])
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.35), Inches(12.1), Inches(4.9), cd).chart
gaya_sumbu(ch, 0.65, 0.80)
pl = label_data(ch)
pl.gap_width, pl.overlap = 60, -10
for sr, w in zip(ch.series, (AKSEN, AKSEN_2)):
    sr.format.fill.solid(); sr.format.fill.fore_color.rgb = w
ch.has_legend = True
ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.TOP, False
ch.legend.font.size = Pt(12)
kotak_teks(s, 0.6, 6.35, 12.1, 0.5, f"Rata-rata 5 fold uji. A0: argmax {f(m('A0','argmax'))}, +bias_val {f(m('A0','bias_val'))}. "
           f"Tertinggi: Mixup argmax {f(m('R-mixup','argmax'))} ± {f(sd('R-mixup','argmax'))}.", 13, warna=ABU)

# ========== 6. Selisih vs A0 ==========
s = slide("Selisih terhadap A0 (poin BACC)")
baris = []
for v in V[1:]:
    a, c = b(v, "argmax"), b(v, "bias_val")
    baris.append([NAMA[v] + (" (terpilih)" if v == terpilih else ""),
                  poin(a["selisih_bacc_rata"]), f"{a['n_fold_positif']}/5", f(a["p_holm"], 3),
                  poin(c["selisih_bacc_rata"]), f"{c['n_fold_positif']}/5", f(c["p_holm"], 3)])
tabel(s, 0.6, 1.5, 12.1, ["Varian", "Δ argmax", "Fold +", "p_holm", "Δ +bias_val", "Fold +", "p_holm"], baris,
      lebar=[3.1, 1.5, 1.4, 1.5, 1.7, 1.4, 1.5], ukuran=15, sorot=V[1:].index(terpilih))
poin_poin(s, 0.6, 4.75, 12.1, 2.0, [
    [("Argmax: ", True), ("Mixup, CutMix, dan stochastic depth positif di 5/5 fold dan melewati 2 poin, tetapi p_holm ≥ 0,35 (n = 5 fold).", False)],
    [("+bias_val: ", True), ("tidak ada varian yang melewati 2 poin. CutMix paling dekat (+1,05 poin, 4/5 fold).", False)],
    [("Label smoothing ", True), ("satu-satunya yang konsisten merugikan (0/5 fold pada argmax).", False)],
], 15)

# ========== 6b. Usulan vs regularisasi ==========
s = slide("Hasil berdampingan: metode usulan vs regularisasi")
assert abs(d3["ringkasan"]["A0|bias_val"]["bacc"]["rata"] - m("A0", "bias_val")) < 1e-9   # A0 identik → sebanding
t3 = d3["a3_terpilih_menurut_validasi"]
r3 = d3["ringkasan"]
MODEL = [
    ("A0 (acuan)", lambda k, met="bacc": m("A0", k, met), 0),
    (f"Derm M3 {t3.removeprefix('A3-')} (usulan, terpilih)", lambda k, met="bacc": r3[f"{t3}|{k}"][met]["rata"], d3["biaya"][t3]["parameter_tambahan"]),
    ("Derm M3 b6-r16 (usulan, uji terbaik)", lambda k, met="bacc": r3[f"A3-b6-r16|{k}"][met]["rata"], d3["biaya"]["A3-b6-r16"]["parameter_tambahan"]),
    ("A0 + CutMix (terpilih)", lambda k, met="bacc": m("R-cutmix", k, met), 0),
    ("A0 + Mixup", lambda k, met="bacc": m("R-mixup", k, met), 0),
]
cd = CategoryChartData()
cd.categories = [n for n, _, _ in MODEL]
cd.add_series("Argmax", [round(g("argmax"), 4) for _, g, _ in MODEL])
cd.add_series("+bias_val", [round(g("bias_val"), 4) for _, g, _ in MODEL])
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.35), Inches(7.4), Inches(5.4), cd).chart
gaya_sumbu(ch, 0.65, 0.80)
pl = label_data(ch)
pl.gap_width, pl.overlap = 60, -10
for sr, w in zip(ch.series, (AKSEN, AKSEN_2)):
    sr.format.fill.solid(); sr.format.fill.fore_color.rgb = w
ch.has_legend = True
ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.TOP, False
ch.legend.font.size = Pt(12)
ch.category_axis.tick_labels.font.size = Pt(11)
tabel(s, 8.3, 1.5, 4.4, ["Model", "Param. +", "AUROC"],
      [[n.split(" (")[0], f"{p:,}".replace(",", "."), f(g("argmax", "auroc_makro"))] for n, g, p in MODEL],
      lebar=[2.2, 1.1, 1.1], ukuran=12)
d_m3 = r3[f"{t3}|bias_val"]["bacc"]["rata"] - m("A0", "bias_val")
d_cm = m("R-cutmix", "argmax") - m("A0", "argmax")
poin_poin(s, 8.3, 4.2, 4.4, 2.6, [
    [("Usulan (M3): ", True), (f"{poin(d_m3)} poin vs A0 (+bias_val); tidak ada konfigurasi M3 yang mengalahkan A0.", False)],
    [("Regularisasi: ", True), (f"CutMix {poin(d_cm)} poin (argmax) tanpa parameter tambahan.", False)],
    [("Split, fold, dan A0 identik ", True), ("di eksperimen 03 dan 05, jadi angka sebanding.", False)],
], 13)

# ========== 6c. Metrik lengkap (F1, presisi, MCC, Brier/MSE, ...) ==========
T5 = json.loads((RUN / "metrik_tambahan.json").read_text())["ringkasan"]
T3 = json.loads((RUN03 / "metrik_tambahan.json").read_text())["ringkasan"]
assert abs(T3["A0|argmax"]["brier"]["rata"] - T5["A0|argmax"]["brier"]["rata"]) < 1e-9
s = slide("Metrik lengkap pada data uji (keputusan argmax)")
BARIS_M = [("A0 (acuan)", T5, R, "A0"), (f"Derm M3 {t3.removeprefix('A3-')} (usulan)", T3, r3, t3)] + \
          [(NAMA[v], T5, R, v) for v in V[1:]]
KOL = [  # (judul, sumber, kunci, makin_besar_makin_baik)
    ("BACC ↑", "r", "bacc", True), ("F1 makro ↑", "t", "f1_makro", True), ("Presisi makro ↑", "t", "presisi_makro", True),
    ("MCC ↑", "t", "mcc", True), ("AUROC ↑", "r", "auroc_makro", True), ("Brier ↓", "t", "brier", False),
    ("MSE ↓", "t", "mse_prob", False), ("Log loss ↓", "t", "log_loss", False), ("ECE ↓", "r", "ece", False),
]


def nilai(t, r, v, src, key):
    return (t[f"{v}|argmax"][key] if src == "t" else r[f"{v}|argmax"][key])["rata"]


isi = [[nilai(t, r, v, src, key) for _, src, key, _ in KOL] for _, t, r, v in BARIS_M]
terbaik = [(max if naik else min)(range(len(isi)), key=lambda i, j=j: isi[i][j]) for j, (_, _, _, naik) in enumerate(KOL)]
tb = tabel(s, 0.6, 1.4, 12.1, ["Model"] + [k[0] for k in KOL],
           [[BARIS_M[i][0]] + [f(x) for x in isi[i]] for i in range(len(isi))],
           lebar=[2.65] + [1.05] * 9, ukuran=13)
for j, i in enumerate(terbaik):           # nilai terbaik per kolom: tebal + latar hijau muda
    c = tb.cell(i + 1, j + 1)
    isi_sel(c, f(isi[i][j]), 13, True, AKSEN, AKSEN_TINT)
kotak_teks(s, 0.6, 5.2, 12.1, 0.7, [
    "Rata-rata 5 fold uji. Tebal/hijau = terbaik per kolom. ↑ makin besar makin baik, ↓ makin kecil makin baik. "
    "Brier = rata-rata Σₖ(pₖ − yₖ)² per citra; MSE = Brier / 7 kelas; keduanya dihitung dari probabilitas.",
], 12, warna=ABU)
poin_poin(s, 0.6, 5.85, 12.1, 1.0, [
    [("F1 makro dan presisi ", True), ("tidak ikut naik seperti BACC: Mixup/CutMix menukar presisi dengan recall kelas minoritas. ", False),
     ("Probabilitas lebih baik: ", True), ("Brier/MSE terendah pada stochastic depth, log loss dan ECE terendah pada CutMix.", False)],
], 14)

# ========== 6d. F1 per kelas ==========
s = slide("F1 per kelas: A0 vs metode usulan vs CutMix")
SER_F1 = [("A0 (acuan)", T5["A0|argmax"], ABU), (f"Derm M3 {t3.removeprefix('A3-')} (usulan)", T3[f"{t3}|argmax"], AKSEN_2),
          ("CutMix", T5["R-cutmix|argmax"], AKSEN)]
cd = CategoryChartData()
cd.categories = KELAS
for n, t, _ in SER_F1:
    cd.add_series(n, [round(t["f1_per_kelas"][k], 4) for k in KELAS])
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.35), Inches(12.1), Inches(4.6), cd).chart
gaya_sumbu(ch, 0, 1.0, "0.00")
pl = label_data(ch, "0.00")
pl.gap_width, pl.overlap = 50, -5
pl.data_labels.font.size = Pt(10)
for sr, (_, _, w) in zip(ch.series, SER_F1):
    sr.format.fill.solid(); sr.format.fill.fore_color.rgb = w
ch.has_legend = True
ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.TOP, False
ch.legend.font.size = Pt(12)
f1a, f1c = T5["A0|argmax"]["f1_per_kelas"], T5["R-cutmix|argmax"]["f1_per_kelas"]
dF1 = sorted(KELAS, key=lambda k: f1c[k] - f1a[k], reverse=True)
kotak_teks(s, 0.6, 6.1, 12.1, 0.8, [
    "Selisih F1 CutMix − A0 (poin): " + " · ".join(f"{k} {poin(f1c[k] - f1a[k])}" for k in dF1),
    f"df: recall naik {poin(recall[terpilih][3] - recall['A0'][3])} poin, tetapi F1 turun — CutMix lebih sering salah menebak df (presisi turun). Rata-rata 5 fold uji, argmax.",
], 13, warna=TEKS)

# ========== 7. Kurva latih ==========
s = slide("Kurva pelatihan: loss latih dan BACC validasi per epoch")
SERI = [("A0", TEKS), ("R-mixup", UNGU), ("R-cutmix", AKSEN), ("R-sd", JINGGA)]
for k, (idx, jdl, mn, mx, fmt) in enumerate(((0, "Loss latih", 0, 1.0, "0.0"), (1, "BACC validasi dalam", 0.30, 0.80, "0.00"))):
    cd = CategoryChartData()
    cd.categories = [str(e) for e in range(1, 21)]
    for v, _ in SERI:
        cd.add_series(NAMA[v], [round(x, 4) for x in kurva[v][idx]])
    ch = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(0.6 + k * 6.25), Inches(1.35), Inches(5.85), Inches(4.2), cd).chart
    gaya_sumbu(ch, mn, mx, fmt)
    judul_grafik(ch, jdl)
    ch.has_legend = True
    ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.BOTTOM, False
    ch.legend.font.size = Pt(12)
    ch.category_axis.has_title = True
    ch.category_axis.axis_title.text_frame.text = "epoch"
    ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
    for sr, (_, w) in zip(ch.series, SERI):
        sr.smooth = False
        sr.format.line.color.rgb = w
        sr.format.line.width = Pt(2.25)
        sr.marker.style = XL_MARKER_STYLE.NONE
A0L, A0B = kurva["A0"]
CML, CMB = kurva["R-cutmix"]
SDL, SDB = kurva["R-sd"]
eA0 = max(range(20), key=lambda i: A0B[i])
eCM = max(range(20), key=lambda i: CMB[i])
poin_poin(s, 0.6, 5.7, 12.1, 1.2, [
    [("A0 menghafal: ", True), (f"loss latih {f(A0L[-1])} di epoch 20; BACC val puncak {f(A0B[eA0],3)} (ep {eA0+1}) lalu turun ke {f(A0B[-1],3)}. ", False),
     ("CutMix: ", True), (f"puncak {f(CMB[eCM],3)} (ep {eCM+1}), akhir {f(CMB[-1],3)}. Pola puncak-lalu-turun masih ada.", False)],
    [("Catatan: ", True), ("loss Mixup/CutMix dihitung terhadap label campuran, jadi tidak sebanding langsung dengan A0. Loss validasi tidak dicatat di run ini.", False)],
], 13)

# ========== 8. Confusion matrix ==========
s = slide(f"Confusion matrix (argmax, gabungan 5 fold uji): A0 vs CutMix")


def heatmap(s, x, y, cm, judul):
    kotak_teks(s, x, y, 5.9, 0.35, judul, 15, True)
    ukuran_sel, kol0 = 0.6, 0.85
    t = s.shapes.add_table(8, 8, Inches(x), Inches(y + 0.4), Inches(kol0 + 7 * ukuran_sel), Inches(0.42 * 8)).table
    t.columns[0].width = Inches(kol0)
    for c in range(1, 8):
        t.columns[c].width = Inches(ukuran_sel)
    for r in range(8):
        t.rows[r].height = Inches(0.42)
    isi_sel(t.cell(0, 0), "asli \\ pred", 10, True, ABU, PUTIH)
    for c, k in enumerate(KELAS):
        isi_sel(t.cell(0, c + 1), k, 11, True, TEKS, PUTIH)
        isi_sel(t.cell(c + 1, 0), k, 11, True, TEKS, PUTIH, PP_ALIGN.LEFT)
    for i in range(7):
        tot = sum(cm[i])
        for j in range(7):
            v = cm[i][j] / tot
            # interpolasi putih → AKSEN
            a = v ** 0.7
            warna = RGBColor(*(int(255 + (c - 255) * a) for c in (0x0F, 0x76, 0x6E)))
            isi_sel(t.cell(i + 1, j + 1), f"{v*100:.0f}", 11, i == j, PUTIH if a > 0.5 else TEKS, warna)


heatmap(s, 0.6, 1.35, CM["A0"], f"A0 (acuan) · BACC uji {f(m('A0','argmax'))}")
heatmap(s, 6.95, 1.35, CM[terpilih], f"CutMix · BACC uji {f(m(terpilih,'argmax'))}")
d_rec = [recall[terpilih][i] - recall["A0"][i] for i in range(7)]
urut = sorted(range(7), key=lambda i: d_rec[i], reverse=True)
teks_rec = " · ".join(f"{KELAS[i]} {poin(d_rec[i])}" for i in urut)
kotak_teks(s, 0.6, 5.25, 12.1, 0.35, "Angka = % baris (recall per kelas asli). Diagonal tebal = prediksi benar. Jumlah citra uji per kelas: "
           + ", ".join(f"{k} {n_kelas[i]:,}".replace(",", ".") for i, k in enumerate(KELAS)) + ".", 12, warna=ABU)
poin_poin(s, 0.6, 5.75, 12.1, 1.1, [
    [("Selisih recall CutMix − A0 (poin): ", True), (teks_rec, False)],
    [("Kelas mel (melanoma): ", True), (f"recall A0 {f(recall['A0'][4]*100,1)}% → CutMix {f(recall[terpilih][4]*100,1)}%; "
     f"mel yang salah ke nv turun {f(CM['A0'][4][5]/sum(CM['A0'][4])*100,1)}% → {f(CM[terpilih][4][5]/sum(CM[terpilih][4])*100,1)}%.", False)],
], 14)

# ========== 9. Kalibrasi ==========
s = slide("Mengapa +bias_val menghapus keunggulan")
cd = CategoryChartData()
cd.categories = [NAMA[v] for v in V]
cd.add_series("ECE (argmax)", [round(m(v, "argmax", "ece"), 4) for v in V])
ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.35), Inches(6.4), Inches(5.3), cd).chart
gaya_sumbu(ch, 0, 0.30)
label_data(ch).gap_width = 50
ch.series[0].format.fill.solid(); ch.series[0].format.fill.fore_color.rgb = AKSEN
ch.has_legend = False
judul_grafik(ch, "ECE, makin kecil makin terkalibrasi")
ch.category_axis.reverse_order = True
gain = {v: m(v, "bias_val") - m(v, "argmax") for v in V}
poin_poin(s, 7.4, 1.5, 5.3, 5.2, [
    [("Efek +bias_val: ", True), (f"A0 {poin(gain['A0'])}, Mixup {poin(gain['R-mixup'])}, stochastic depth {poin(gain['R-sd'])}, CutMix {poin(gain['R-cutmix'])} poin.", False)],
    [("Bias overfit ke validasi: ", True), (f"BACC val +bias_val CutMix {f(m('R-cutmix','bias_val','bacc_val'))}, tetapi uji {f(m('R-cutmix','bias_val'))}.", False)],
    [("Dugaan: ", True), ("A0 terlalu yakin, jadi koreksi bias membantu. Model yang diregularisasi sudah terkalibrasi; yang tersisa dari bias hanya noise. Belum diuji formal.", False)],
], 15)

# ========== 10. Kesimpulan ==========
s = slide("Kesimpulan dan langkah berikutnya")
kotak_teks(s, 0.6, 1.4, 5.8, 0.4, "Kesimpulan", 20, True)
poin_poin(s, 0.6, 1.95, 5.8, 4.8, [
    "Menurut aturan pra-registrasi: hipotesis tidak didukung.",
    "Cara latih lebih berpengaruh daripada arsitektur: tiga teknik menang di 5/5 fold (argmax).",
    "Label smoothing merugikan; tukar latar netral.",
    "Temuan argmax muncul dari analisis sekunder, jadi belum boleh diklaim.",
], 16)
kotak_teks(s, 7.0, 1.4, 5.7, 0.4, "Usulan eksperimen 06", 20, True)
kt = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.0), Inches(1.95), Inches(5.7), Inches(3.6))
kt.fill.solid(); kt.fill.fore_color.rgb = LATAR_KARTU; kt.line.fill.background(); kt.shadow.inherit = False
poin_poin(s, 7.3, 2.2, 5.1, 3.2, [
    "Hipotesis dikunci dulu: Mixup / CutMix / stochastic depth (argmax) vs A0 +bias_val.",
    "Split baru: seed 1 dan 2.",
    "Ikut diuji: kombinasi CutMix + stochastic depth.",
    "Catat loss validasi per epoch.",
], 15)

prs.save(OUT)
print("tersimpan", OUT, "slide:", len(prs.slides))
print("recall A0", [round(x, 3) for x in recall["A0"]])
print("recall", terpilih, [round(x, 3) for x in recall[terpilih]])
print("n kelas", n_kelas)
