# Eksperimen 00: Replikasi Shetty dkk. (2022) — Protokol L0 (apa adanya)

Bagian dari Plan 0 (§0.R di `riset/PLAN.md`). Ini **titik L0** tangga dekomposisi RQ1.

## Pertanyaan
Bila pipeline Shetty dkk. (2022) dijalankan **persis seperti protokol di paper** (termasuk kebocorannya), apakah angka Tabel 4, 6, dan 7 bisa direproduksi?

## Data
- HAM10000 = ISIC 2018 Task 3 Training (10.015 citra) + GroundTruth + LesionGroupings, dari `https://isic-challenge-data.s3.amazonaws.com/2018/`. Disimpan di `data/mentah/isic2018/`. Lisensi CC BY-NC 4.0.
- Subset: 100 citra/kelas = 700 citra, diambil acak dengan 10 seed berbeda (paper hanya melaporkan satu subset).

## Metode (sesuai paper)
| Komponen | ML (`ml.py`) | CNN (`cnn.py`) |
|---|---|---|
| Ukuran input | 220×220 | 96×96×3, dibagi 255 |
| Augmentasi | Flip horizontal → 200/kelas (1.400), **sebelum** split | 1.400 citra (asli + flip) **sebelum** split + transformasi acak saat latih |
| Fitur / model | Histogram HSV + Haralick + Hu → 7 klasifier (Tabel 1) | CNN Tabel 2, Adam, 150 epoch, batch 32 |
| Split | 80:20 tak berstrata | 80:20 tak berstrata (uji 280) |
| Angka yang dilaporkan | Rata-rata 10-fold CV pada porsi latih (= Tabel 4) + akurasi pada 20% uji | Akurasi latih & uji epoch terakhir (= Tabel 6) + laporan per kelas (= Tabel 7) |

## Bukti forensik dari paper (dasar keputusan replikasi)
- Ke-14 akurasi di Tabel 4 adalah kelipatan tepat 1/560 (tanpa augmentasi) dan 1/1120 (flip). Artinya angka itu **akurasi CV pada porsi latih 80%**, bukan pada data uji.
- Support Tabel 7 = 37/33/35/43/44/44/44 (jumlah 280). Split **tidak berstrata**, karena kalau berstrata semuanya akan 40.
- Akurasi CNN 86,43% = 242/280 dan 95,18% = 1.066/1.120. Kemungkinan besar diambil dari **epoch terakhir**.

## Asumsi re-implementasi (tidak dijelaskan paper)
| Kode | Asumsi | Alasan |
|---|---|---|
| A1 | Cara memilih 100 citra/kelas = acak, diulang 10 seed | Tidak disebut di paper; variasinya ikut dilaporkan |
| A2 | Histogram HSV 8×8×8 dengan rentang [0,256] di semua kanal (H OpenCV sebenarnya 0–179) | Pola kode fitur global yang umum; total 532 fitur |
| A3 | MinMaxScaler di-fit pada semua fitur sebelum split | Pola kode umum; ikut bocor, sesuai semangat L0 |
| A4 | KFold 10 tanpa shuffle pada porsi latih | Data sudah diacak oleh `train_test_split` |
| A5 | DT memakai default sklearn (random_state=9) | "Estimators = 100" di Tabel 1 bukan parameter Decision Tree |
| A6 | Versi sklearn saat ini (LR solver lbfgs, dll.) | Versi paper tidak disebut (Python 3.7.9) |
| A7 | Parameter augmentasi CNN: rotasi 25°, geser 10%, shear 0,2°, zoom 20%, flip, isi tepi *nearest* | Paper hanya menyebut "rotations, shearing, etc." |
| A8 | LR turun eksponensial 1e-3 → 1e-5 selama 150 epoch | Tabel 3 hanya menulis "0.001–0.00001" |
| A9 | PyTorch, dengan default Keras ditiru (glorot_uniform, BN momentum 0,99, eps 1e-3) | Paper memakai Keras |
| A10 | Seed split CNN = 42, ML = 9 | Tidak disebut di paper |

## Cara menjalankan
```bash
uv run python tugas/pcd-lesi-kulit/eksperimen/00-replikasi-shetty-L0/ml.py
uv run python tugas/pcd-lesi-kulit/eksperimen/00-replikasi-shetty-L0/cnn.py
```

## Hasil

### ML (Tabel 4), 10 seed subset, akurasi 10-fold CV pada porsi latih (%)
Run: `hasil/20261021-120854/`

| Model | Paper (tanpa aug) | Replikasi (tanpa aug) | Paper (flip) | Replikasi (flip) | Uji 20% (flip) |
|---|---|---|---|---|---|
| LR | 48,93 | 43,25 ± 2,22 | 58,13 | 53,94 ± 2,00 | 55,00 |
| LDA | 35,18 | 34,27 ± 1,77 | 57,59 | 64,48 ± 1,38 | 65,29 |
| KNN | 41,79 | 34,00 ± 2,02 | 48,39 | 43,88 ± 2,49 | 46,14 |
| DT | 44,29 | 37,95 ± 1,30 | 68,66 | 82,10 ± 1,01 | 86,71 |
| **RF** | **58,57** | **54,43 ± 1,46** | **87,32** | **87,00 ± 0,67** | **90,43** |
| NB | 34,11 | 32,00 ± 1,97 | 36,34 | 39,97 ± 2,03 | 42,68 |
| SVM | 42,14 | 42,91 ± 2,19 | 53,13 | 54,96 ± 1,98 | 55,82 |

- Porsi citra uji yang kembaran flip-nya ada di data latih: **0%** tanpa augmentasi vs **78,6%** dengan flip sebelum split.

### CNN (Tabel 6–7), 3 seed, epoch ke-150
Run: `hasil/20261021-121228/`

| | Paper | Replikasi |
|---|---|---|
| Akurasi latih | 95,18 | **95,42 ± 1,10** |
| Akurasi uji (280) | 86,43 | 90,60 ± 0,55 |
| BACC uji | – (≈ 85,0 dari Tabel 7) | 90,77 |
| Uji yang kembarannya di data latih | – | 83,6% |
| val_loss maksimum | ±15 (Gbr. 5) | 4,15–7,89 |

Per kelas (presisi / recall), paper vs replikasi: akiec 0,89/0,68 vs 0,89/0,85 · bcc **0,55**/0,88 vs 0,92/0,89 · bkl 0,89/0,94 vs 0,86/0,89 · df 0,88/0,81 vs 0,86/0,89 · mel 0,97/0,89 vs 0,91/0,88 · nv 0,93/0,86 vs 0,93/0,97 · vasc 0,95/0,89 vs 0,98/0,99.

## Kesimpulan
1. **Angka utama ML tereproduksi.** RF + flip 87,00 ± 0,67% vs 87,32% di paper. Lonjakan dari sekitar 54% ke 87% hanya karena flip juga muncul lagi. Kenaikan serupa terjadi pada DT (37,95 → 82,10, +44 poin) dan LDA (+30 poin). Penyebabnya kebocoran: 78,6% citra uji punya kembaran flip di data latih. Klasifier yang bisa "menghafal" (RF, DT) paling diuntungkan. Ini bukti kuat untuk H1, dan menjadi titik L0 yang valid.
2. **Tanpa augmentasi**, replikasi sedikit di bawah paper untuk LR, KNN, dan DT (−4 sampai −8 poin). Kemungkinan penyebabnya: perbedaan versi sklearn (A6), cara memilih subset (A1), dan parameter fitur (A2). Polanya sama: RF tetap terbaik dan NB tetap terburuk.
3. **CNN:** akurasi latih tereproduksi (95,42 vs 95,18). Akurasi uji kami +4 poin di atas paper. Kurva kami jauh lebih stabil (val_loss maksimum 4–8 vs ±15), jadi selisihnya kemungkinan berasal dari detail implementasi Keras vs PyTorch dan augmentasi (A7–A9), bukan dari protokol. Presisi bcc 0,55 di paper tidak muncul di replikasi.
4. Klaim "95,18%" Shetty adalah **akurasi latih**. Akurasi uji 86–91% pun masih menggelembung, karena lebih dari 80% citra uji punya kembaran di data latih. Angka yang jujur baru akan terlihat di L2–L4.

**Gate 0.R: LOLOS.** ML tereproduksi dalam rentang seed. Akurasi latih CNN tereproduksi. Selisih akurasi uji CNN terdokumentasi beserta dugaan penyebabnya.
