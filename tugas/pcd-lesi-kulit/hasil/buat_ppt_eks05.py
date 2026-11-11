"""Deck sederhana hasil eksperimen 05. Semua angka dibaca dari metrics.json / per_run."""
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
d = json.loads((RUN / "metrics.json").read_text())
R, B = d["ringkasan"], d["banding_a3_vs_A0"]
V = ["A0", "R-ls", "R-mixup", "R-cutmix", "R-sd", "R-10ep", "R-latar"]
NAMA = {"A0": "A0 (acuan)", "R-ls": "Label smoothing", "R-mixup": "Mixup", "R-cutmix": "CutMix",
        "R-sd": "Stochastic depth", "R-10ep": "10 epoch", "R-latar": "Tukar latar"}
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

# diagnostik latih
diag = {}
for v in V:
    H = [json.loads((RUN / f"per_run/{v}_r0_f{i}.json").read_text())["riwayat"] for i in range(5)]
    n = len(H[0])
    L = [sum(h[e]["loss"] for h in H) / 5 for e in range(n)]
    Bv = [sum(h[e]["bacc_val"] for h in H) / 5 for e in range(n)]
    e = max(range(n), key=lambda i: Bv[i])
    diag[v] = (n, L[-1], Bv[e], e + 1, Bv[-1])

# ---------- gaya ----------
TEKS = RGBColor(0x1F, 0x29, 0x37)
ABU = RGBColor(0x6B, 0x72, 0x80)
ABU_MUDA = RGBColor(0xE5, 0xE7, 0xEB)
LATAR_KARTU = RGBColor(0xF3, 0xF4, 0xF6)
AKSEN = RGBColor(0x0F, 0x76, 0x6E)
AKSEN_2 = RGBColor(0x99, 0xC9, 0xC3)
MERAH = RGBColor(0xB9, 0x1C, 0x1C)
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
    baris = teks if isinstance(teks, list) else [teks]
    for i, t in enumerate(baris):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(8)
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
        bagian = it if isinstance(it, list) else [(it, False)]
        r0 = p.add_run()
        r0.text = "•  "
        r0.font.size, r0.font.name = Pt(ukuran), FONT
        r0.font.color.rgb = AKSEN
        for t, tebal in bagian:
            r = p.add_run()
            r.text = t
            r.font.size, r.font.bold, r.font.name = Pt(ukuran), tebal, FONT
            r.font.color.rgb = TEKS

def slide(judul):
    s = prs.slides.add_slide(KOSONG)
    nomor[0] += 1
    kotak_teks(s, 0.6, 0.45, 12.1, 0.8, judul, 30, True)
    kotak_teks(s, 0.6, 7.0, 9, 0.3, "Eksperimen 05 · Regularisasi pelatihan MobileNetV3-Large · HAM10000", 10, warna=ABU)
    kotak_teks(s, 11.7, 7.0, 1.0, 0.3, str(nomor[0]), 10, warna=ABU, align=PP_ALIGN.RIGHT)
    return s

def tabel(s, x, y, w, kolom, baris, lebar=None, ukuran=13, sorot=None):
    t = s.shapes.add_table(len(baris) + 1, len(kolom), Inches(x), Inches(y), Inches(w), Inches(0.4 * (len(baris) + 1))).table
    if lebar:
        for i, lw in enumerate(lebar):
            t.columns[i].width = Inches(lw)
    for r in range(len(baris) + 1):
        t.rows[r].height = Inches(0.4)
        for c in range(len(kolom)):
            cell = t.cell(r, c)
            teks = kolom[c] if r == 0 else baris[r - 1][c]
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = teks
            run.font.size, run.font.name = Pt(ukuran), FONT
            run.font.bold = r == 0 or (sorot is not None and r - 1 == sorot)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if r == 0 else TEKS
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = TEKS
            elif sorot is not None and r - 1 == sorot:
                cell.fill.fore_color.rgb = RGBColor(0xDC, 0xEF, 0xEC)
            else:
                cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if r % 2 else LATAR_KARTU
    return t

def gaya_grafik(ch, minimum, maksimum, fmt="0.000"):
    ch.font.size, ch.font.name = Pt(12), FONT
    ch.font.color.rgb = TEKS
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = minimum, maksimum
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = ABU_MUDA
    va.format.line.fill.background()
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = fmt, False
    ch.category_axis.format.line.color.rgb = ABU_MUDA
    pl = ch.plots[0]
    pl.has_data_labels = True
    dl = pl.data_labels
    dl.number_format, dl.number_format_is_linked = fmt, False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(11)
    return pl

# ---------- 1. Judul ----------
s = prs.slides.add_slide(KOSONG)
nomor[0] += 1
kotak_teks(s, 0.8, 2.3, 11.7, 1.4, "Regularisasi Pelatihan pada MobileNetV3-Large untuk Klasifikasi Lesi Kulit", 38, True)
kotak_teks(s, 0.8, 3.9, 11.7, 0.6, "Eksperimen 05 · HAM10000 · 7 varian × 5 fold per lesi", 20, warna=AKSEN)
kotak_teks(s, 0.8, 5.6, 11.7, 0.9, ["Pengolahan Citra Digital · Magister Informatika ITS",
                                     f"Run hasil/20261109-164154 · draf, {f(d['durasi_jam'], 2)} jam GPU"], 14, warna=ABU)

# ---------- 2. Ringkasan ----------
s = slide("Ringkasan: menang di argmax, belum lolos aturan utama")
kartu = [
    (poin(b("R-mixup", "argmax")["selisih_bacc_rata"]) + " poin",
     f"Mixup vs A0, keputusan argmax\nunggul di {b('R-mixup','argmax')['n_fold_positif']}/5 fold · p_holm {f(b('R-mixup','argmax')['p_holm'],3)}"),
    (poin(b(terpilih, "bias_val")["selisih_bacc_rata"]) + " poin",
     f"CutMix (terpilih menurut validasi), +bias_val\np_holm {f(b(terpilih,'bias_val')['p_holm'],3)} · di bawah ambang 2 poin"),
    (f"{f(m('A0','argmax','ece'))} → {f(m('R-cutmix','argmax','ece'))}",
     "ECE A0 → CutMix\nmodel jauh lebih terkalibrasi"),
]
for i, (angka, ket) in enumerate(kartu):
    x = 0.6 + i * 4.1
    kt = s.shapes.add_shape(1, Inches(x), Inches(1.6), Inches(3.8), Inches(2.6))
    kt.fill.solid(); kt.fill.fore_color.rgb = LATAR_KARTU; kt.line.fill.background(); kt.shadow.inherit = False
    kotak_teks(s, x + 0.3, 1.9, 3.2, 0.9, angka, 36, True, AKSEN if i != 1 else TEKS)
    kotak_teks(s, x + 0.3, 2.95, 3.2, 1.1, ket.split("\n"), 14, warna=TEKS)
poin_poin(s, 0.6, 4.6, 12.1, 2.2, [
    [("Hipotesis \"regularisasi > A0\" tidak didukung ", True), ("menurut aturan yang ditetapkan sebelum run.", False)],
    [("Mixup, CutMix, dan stochastic depth ", True), ("mengalahkan A0 di semua 5 fold pada argmax (+2,6 s.d. +3,4 poin), tetapi belum bermakna setelah koreksi Holm.", False)],
    [("Penyetelan bias per kelas ", True), ("membantu A0, tetapi menghapus keunggulan model yang sudah diregularisasi.", False)],
], 16)

# ---------- 3. Desain ----------
s = slide("Desain eksperimen")
kotak_teks(s, 0.6, 1.4, 5.6, 0.4, "Mengapa cara latih, bukan arsitektur", 18, True)
poin_poin(s, 0.6, 1.95, 5.6, 4.8, [
    "16 varian modifikasi arsitektur (eksperimen 01–04) tidak ada yang mengalahkan A0.",
    f"A0 menghafal data latih: loss latih {f(diag['A0'][1])} di epoch 20.",
    f"BACC validasi A0 memuncak di epoch {diag['A0'][3]}, lalu turun.",
    "Semua varian = arsitektur A0, hiperparameter bawaan paper, tanpa tuning.",
    "Data: HAM10000, StratifiedGroupKFold 5 fold per lesi (seed 0), validasi dalam 10% lesi.",
    "Varian terpilih = BACC validasi tertinggi; menang bila ≥ 2 poin dan p_holm < 0,05 (12 perbandingan).",
], 15)
tabel(s, 6.7, 1.4, 6.0, ["Varian", "Perubahan"], [
    ["Label smoothing", "ε = 0,1"],
    ["Mixup", "α = 0,2"],
    ["CutMix", "α = 1,0"],
    ["Stochastic depth", "p naik linear s.d. 0,2"],
    ["10 epoch", "jadwal cosine 10 (bukan 20) epoch"],
    ["Tukar latar", "lesi ditempel ke kulit citra lain, p = 0,5"],
], lebar=[2.2, 3.8], ukuran=14)

# ---------- 4. Grafik BACC ----------
s = slide("BACC uji: regularisasi naik di argmax, tidak di +bias_val")
cd = CategoryChartData()
cd.categories = [NAMA[v] for v in V]
cd.add_series("Argmax", [round(m(v, "argmax"), 4) for v in V])
cd.add_series("+bias_val (keputusan utama)", [round(m(v, "bias_val"), 4) for v in V])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.35), Inches(12.1), Inches(4.9), cd)
ch = gf.chart
pl = gaya_grafik(ch, 0.65, 0.80)
pl.gap_width, pl.overlap = 60, -10
ch.series[0].format.fill.solid(); ch.series[0].format.fill.fore_color.rgb = AKSEN
ch.series[1].format.fill.solid(); ch.series[1].format.fill.fore_color.rgb = AKSEN_2
ch.has_legend = True
ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.TOP, False
ch.legend.font.size = Pt(12)
kotak_teks(s, 0.6, 6.35, 12.1, 0.5, f"Rata-rata 5 fold uji. A0: argmax {f(m('A0','argmax'))}, +bias_val {f(m('A0','bias_val'))}. "
           f"Tertinggi: Mixup argmax {f(m('R-mixup','argmax'))} ± {f(sd('R-mixup','argmax'))}.", 13, warna=ABU)

# ---------- 5. Selisih vs A0 ----------
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

# ---------- 6. Metrik lengkap ----------
s = slide("Metrik lengkap (keputusan argmax)")
baris = [[NAMA[v], f(m(v, "argmax", "bacc_val")), f"{f(m(v,'argmax'))} ± {f(sd(v,'argmax'))}",
          f(m(v, "argmax", "auroc_makro")), f(m(v, "argmax", "f1_makro")),
          f(m(v, "argmax", "sens_mel")), f(m(v, "argmax", "ece"))] for v in V]
tabel(s, 0.6, 1.5, 12.1, ["Varian", "BACC val", "BACC uji", "AUROC makro", "F1 makro", "Sens. MEL", "ECE"], baris,
      lebar=[2.6, 1.4, 2.3, 1.6, 1.4, 1.4, 1.4], ukuran=15, sorot=V.index(terpilih))
kotak_teks(s, 0.6, 5.0, 12.1, 1.4, [
    "Biaya inferensi semua varian identik dengan A0: 4.210.999 parameter, 430,6 MFLOPs. Regularisasi hanya aktif saat latih.",
    "AUROC naik pada stochastic depth, CutMix, dan tukar latar dibanding A0; selisih AUROC tidak diuji statistik.",
], 14, warna=ABU)

# ---------- 7. Kalibrasi ----------
s = slide("Mengapa +bias_val menghapus keunggulan")
cd = CategoryChartData()
cd.categories = [NAMA[v] for v in V]
cd.add_series("ECE (argmax)", [round(m(v, "argmax", "ece"), 4) for v in V])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.35), Inches(6.4), Inches(5.3), cd)
ch = gf.chart
pl = gaya_grafik(ch, 0, 0.30)
pl.gap_width = 50
ch.series[0].format.fill.solid(); ch.series[0].format.fill.fore_color.rgb = AKSEN
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = "ECE, makin kecil makin terkalibrasi"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
ch.category_axis.reverse_order = True
gain = {v: m(v, "bias_val") - m(v, "argmax") for v in V}
poin_poin(s, 7.4, 1.5, 5.3, 5.2, [
    [("Efek +bias_val: ", True), (f"A0 {poin(gain['A0'])}, Mixup {poin(gain['R-mixup'])}, stochastic depth {poin(gain['R-sd'])}, CutMix {poin(gain['R-cutmix'])} poin.", False)],
    [("Bias overfit ke validasi: ", True), (f"BACC val +bias_val CutMix {f(m('R-cutmix','bias_val','bacc_val'))}, tetapi uji {f(m('R-cutmix','bias_val'))}.", False)],
    [("Dugaan: ", True), ("A0 terlalu yakin, jadi koreksi bias membantu. Model yang diregularisasi sudah terkalibrasi; yang tersisa dari bias hanya noise. Belum diuji formal.", False)],
], 15)

# ---------- 8. Hafalan ----------
s = slide("Apakah masalah hafalan berkurang? Sebagian")
baris = [[NAMA[v], f(diag[v][1]), f"{f(diag[v][2],3)} (ep {diag[v][3]})", f(diag[v][4], 3)] for v in V]
tabel(s, 0.6, 1.5, 7.4, ["Varian", "Loss latih akhir", "BACC val tertinggi", "BACC val akhir"], baris,
      lebar=[2.3, 1.6, 2.0, 1.5], ukuran=14)
poin_poin(s, 8.4, 1.5, 4.3, 5.2, [
    "Loss latih tidak lagi ≈ 0,02 dan BACC validasi tertinggi naik.",
    "Pola puncak di epoch 12–13 lalu turun masih ada; hanya tukar latar yang memuncak di epoch akhir.",
    "Epoch pendek saja tidak menggantikan regularisasi.",
], 15)
kotak_teks(s, 0.6, 5.0, 7.4, 1.0, "Rata-rata 5 fold. Loss label smoothing, mixup, dan CutMix dihitung terhadap target yang dilunakkan/dicampur, jadi tidak sebanding langsung dengan A0. Untuk varian 10 epoch, “akhir” = epoch 10.", 12, warna=ABU)

# ---------- 9. Kesimpulan ----------
s = slide("Kesimpulan dan langkah berikutnya")
kotak_teks(s, 0.6, 1.4, 5.8, 0.4, "Kesimpulan", 20, True)
poin_poin(s, 0.6, 1.95, 5.8, 4.8, [
    "Menurut aturan pra-registrasi: hipotesis tidak didukung.",
    "Cara latih lebih berpengaruh daripada arsitektur: tiga teknik menang di 5/5 fold (argmax).",
    "Label smoothing merugikan; tukar latar netral.",
    "Temuan argmax muncul dari analisis sekunder, jadi belum boleh diklaim.",
], 16)
kotak_teks(s, 7.0, 1.4, 5.7, 0.4, "Usulan eksperimen 06", 20, True)
kt = s.shapes.add_shape(1, Inches(7.0), Inches(1.95), Inches(5.7), Inches(3.6))
kt.fill.solid(); kt.fill.fore_color.rgb = LATAR_KARTU; kt.line.fill.background(); kt.shadow.inherit = False
poin_poin(s, 7.3, 2.2, 5.1, 3.2, [
    "Hipotesis dikunci dulu: Mixup / CutMix / stochastic depth (argmax) vs A0 +bias_val.",
    "Split baru: seed 1 dan 2.",
    "Ikut diuji: kombinasi CutMix + stochastic depth.",
    "Tukar latar diuji ulang dengan protokol H5d.",
], 15)

prs.save(OUT)
print("tersimpan", OUT, "slide:", len(prs.slides))
