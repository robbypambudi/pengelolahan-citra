# Eksperimen 05: cara latih, bukan arsitektur (regularisasi pada A0)

Lanjutan dari eksperimen 03 dan 04 (`../04-attention/`, run `hasil/20261109-123813`).

Kode:
- `citra/regularisasi.py`, dengan tes di `tests/test_regularisasi.py`
- fungsi `latih` di `run.py`
- alur lain memakai ulang `../03-derm-v2/run.py`

## Latar belakang
Di eksperimen 01–04, 16 varian modifikasi arsitektur tidak ada yang mengalahkan A0. Polanya selalu sama:
- *loss* latih ≈ 0,02 di epoch 20, artinya data latih hampir dihafal;
- BACC validasi memuncak di epoch 10–12, lalu turun.

Dugaannya, hambatan ada pada **generalisasi**, bukan kapasitas. Karena itu eksperimen ini mempertahankan arsitektur A0 dan hanya mengubah cara melatihnya.

## Pertanyaan
Apakah regularisasi pelatihan menaikkan BACC A0 pada protokol per lesi yang sama? Apakah augmentasi **tukar latar** (dimotivasi H5d) membantu?

## Aturan (ditetapkan SEBELUM run penuh, 9 Nov 2026)
1. Arsitektur semua varian = A0. Strategi latih: S6-logit-adj. Keputusan utama: +bias_val.
2. Hiperparameter setiap teknik = **nilai bawaan dari paper asalnya, tanpa tuning**. Tukar latar memakai p = 0,5 dan dilasi 16 px (sama dengan A0-latar di eksperimen 03).
3. **Varian terpilih** = BACC validasi dalam rata-rata tertinggi (argmax) di antara 6 varian. Fold uji tidak dipakai untuk memilih.
4. Hipotesis "regularisasi > A0" **didukung** bila varian terpilih unggul ≥ SESOI 2 poin BACC (+bias_val) dengan p_holm < 0,05. Koreksi Holm diterapkan atas 12 perbandingan (6 varian × 2 keputusan).
5. Bila didukung, kemenangan **wajib dikonfirmasi** dengan split baru (seed ulangan 1 dan 2). Kombinasi teknik (mis. mixup + tukar latar) juga baru diuji di tahap konfirmasi, bukan dipilih dari fold ini.
6. Semua varian dilaporkan. A0 harus sama persis dengan eksperimen 02–04.

## Data
Sama dengan eksperimen 01–04:
- cache 256 px
- StratifiedGroupKFold 5 fold per lesi, seed 0
- validasi dalam 10% lesi

## Metode
| Kode | Perubahan | Nilai | Rujukan |
|---|---|---|---|
| A0 | – | – | acuan |
| R-ls | *label smoothing* | ε = 0,1 | Szegedy dkk. (2016) |
| R-mixup | mixup setiap batch | α = 0,2 | Zhang dkk. (2018) |
| R-cutmix | cutmix setiap batch | α = 1,0 | Yun dkk. (2019) |
| R-sd | *stochastic depth* pada blok residual | p naik linear sampai 0,2 | Huang dkk. (2016) |
| R-10ep | jadwal cosine 10 epoch (bukan 20) | – | dari kurva validasi eksperimen 03/04 |
| R-latar | **tukar latar**: lesi + 16 px tepinya ditempel ke kulit citra lain yang lesinya sudah dihapus (diisi rata-rata ternormalisasi kulit sekitar); label tetap label lesi | 50% sampel per batch | usulan kami, dimotivasi H5d dan Bissoto dkk. (2019) |

Detail implementasi:
- *Logit-adjusted loss* tetap dipakai. Untuk mixup dan cutmix: loss = λ·L(y) + (1 − λ)·L(y_perm).
- Regularisasi memakai generator acak terpisah, sehingga urutan batch dan augmentasi geometri sama persis dengan A0.
- Semua teknik hanya aktif saat latih. Evaluasi sama dengan A0.

## Keterbatasan yang diketahui sebelum run
- Tukar latar membawa cincin kulit asli selebar ±16 px di sekitar lesi. Ini sengaja, agar halo pigmen yang tidak tertangkap mask Otsu tidak terpotong. Akibatnya, sebagian kecil informasi latar asli tetap ada.
- Warna kulit donor dan penerima bisa berbeda, sehingga tepinya terlihat.
- Pada 439 citra, mask Otsu adalah elips cadangan.
- Tanpa tuning, teknik yang sensitif terhadap hiperparameter (mixup/cutmix pada *fine-tuning* singkat) bisa dirugikan.

## Cara menjalankan
```bash
uv run python tugas/pcd-lesi-kulit/eksperimen/05-regularisasi/run.py --config config-cepat.yaml   # uji jalan
uv run python tugas/pcd-lesi-kulit/eksperimen/05-regularisasi/run.py                               # penuh, ±1,3 jam
```

## Hasil
| Run | Mode | Catatan |
|---|---|---|
| `hasil/20261109-164105` | cepat (1 fold, 2 epoch, 1.500 citra; R-10ep = 1 epoch) | Uji jalan; angkanya **tidak** untuk laporan. A0 lewat fungsi latih baru = A0 eksperimen 02–04, identik (0,2029 / 0,3446). |
| `hasil/20261109-164154` | penuh, 1×5 fold | Run acuan. 1,17 jam. Basis commit `9eca953` + perubahan belum di-commit (`perubahan_belum_commit.patch`). |

### Run `hasil/20261109-164154` (5 fold uji, rata-rata ± SD)
Cek konsistensi: A0 = 0,7388 / 0,7522, sama persis dengan eksperimen 02–04. Biaya inferensi semua varian identik dengan A0 (4.210.999 parameter, 430,6 MFLOPs).

| Varian | BACC val (argmax) | BACC uji argmax | BACC uji +bias_val | Δ vs A0 (+bias) | Fold positif | p_holm | AUROC makro | F1 makro (argmax) | ECE (argmax) |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 0,7407 | 0,7388 ± 0,0212 | **0,7522 ± 0,0230** | – | – | – | 0,9575 | 0,6945 | 0,1079 |
| R-ls | 0,7252 | 0,7019 ± 0,0174 | 0,7337 ± 0,0263 | −0,0185 | 1/5 | 1,000 | 0,9176 | 0,5278 | 0,2356 |
| R-mixup | 0,7449 | **0,7726 ± 0,0157** | 0,7532 ± 0,0320 | +0,0010 | 3/5 | 1,000 | 0,9438 | 0,6684 | 0,0734 |
| **R-cutmix** (terpilih) | **0,7795** | 0,7648 ± 0,0154 | 0,7627 ± 0,0110 | +0,0105 | 4/5 | 1,000 | 0,9627 | 0,6607 | **0,0185** |
| R-sd | 0,7633 | 0,7673 ± 0,0173 | 0,7477 ± 0,0161 | −0,0045 | 2/5 | 1,000 | **0,9650** | 0,6895 | 0,0771 |
| R-10ep | 0,7365 | 0,7419 ± 0,0249 | 0,7449 ± 0,0263 | −0,0073 | 1/5 | 1,000 | 0,9593 | 0,6735 | 0,0758 |
| R-latar | 0,7475 | 0,7510 ± 0,0162 | 0,7461 ± 0,0149 | −0,0061 | 3/5 | 1,000 | 0,9624 | 0,6941 | 0,0823 |

Selisih argmax vs A0 (fold positif, p_holm):

| Varian | Δ | Fold positif | p_holm |
|---|---|---|---|
| R-ls | −0,0369 | 0/5 | 0,060 |
| R-mixup | +0,0338 | 5/5 | 0,382 |
| R-cutmix | +0,0261 | 5/5 | 0,352 |
| R-sd | +0,0285 | 5/5 | 0,769 |
| R-10ep | +0,0031 | 3/5 | 1,000 |
| R-latar | +0,0122 | 4/5 | 1,000 |

Diagnostik latih (rata-rata 5 fold). *Loss* R-ls, R-mixup, dan R-cutmix dihitung terhadap target yang dilunakkan/dicampur, sehingga tidak sebanding langsung dengan A0.

| Varian | *Loss* latih epoch akhir | BACC val tertinggi | BACC val epoch akhir |
|---|---|---|---|
| A0 | 0,0225 | 0,749 (ep 12) | 0,741 |
| R-ls | 0,4813 | 0,726 (ep 17) | 0,725 |
| R-mixup | 0,2689 | 0,756 (ep 13) | 0,745 |
| R-cutmix | 0,5586 | 0,787 (ep 12) | 0,779 |
| R-sd | 0,1574 | 0,770 (ep 13) | 0,763 |
| R-10ep | 0,1019 (ep 10) | 0,737 (ep 6) | 0,737 |
| R-latar | 0,1440 | 0,748 (ep 20) | 0,748 |

## Kesimpulan
**Menurut aturan yang ditetapkan sebelumnya, hipotesis "regularisasi > A0" tidak didukung.**
- Varian terpilih menurut validasi adalah R-cutmix. Pada keputusan utama (+bias_val), R-cutmix unggul +1,05 poin atas A0 (positif di 4/5 fold, p_holm 1,000). Ini di bawah SESOI 2 poin dan tidak bermakna.
- Karena hipotesis tidak didukung, konfirmasi dengan seed split 1 dan 2 (aturan 5) tidak diwajibkan.

**Temuan utama: regularisasi menaikkan BACC argmax, tetapi keuntungannya hilang setelah penyetelan bias.**
- Pada keputusan argmax, tiga varian unggul atas A0 di **semua 5 fold**:

  | Varian | BACC uji argmax | Δ vs A0 |
  |---|---|---|
  | R-mixup | 0,7726 | +3,38 poin |
  | R-sd | 0,7673 | +2,85 poin |
  | R-cutmix | 0,7648 | +2,61 poin |

  Ketiganya melampaui SESOI dan bahkan lebih tinggi daripada A0 +bias_val (0,7522), tetapi setelah koreksi Holm atas 12 perbandingan tidak bermakna (p_holm 0,35–0,77; n = 5 fold).
- Penyetelan bias per kelas di validasi dalam, yang menaikkan A0 sebesar +1,3 poin, justru **menurunkan** varian yang sudah diregularisasi:
  - R-mixup −1,9 poin;
  - R-sd −2,0 poin;
  - R-cutmix −0,2 poin.

  Selisih BACC validasi +bias_val (≈ 0,79–0,83) dengan BACC uji (≈ 0,75–0,76) menunjukkan bias itu *overfit* ke validasi dalam (10% lesi).
- Dugaan kami, bias_val berguna bagi A0 karena keluaran A0 terlalu yakin (ECE 0,1079). Model yang diregularisasi lebih terkalibrasi (ECE R-cutmix 0,0185), sehingga yang tersisa dari penyetelan bias hanya *noise*. Dugaan ini belum diuji secara formal.

**Apakah masalah hafalan berkurang?**
- Sebagian. *Loss* latih tidak lagi ≈ 0,02. Misalnya R-sd 0,157 dan R-latar 0,144, yang dihitung terhadap target asli.
- BACC validasi tertinggi naik, misalnya R-cutmix 0,787 vs A0 0,749.
- Pola puncak di epoch 12–13 lalu turun ≈ 0,8 poin tetap ada. Hanya R-latar yang tidak turun (puncaknya di epoch 20).
- Epoch yang lebih pendek (R-10ep) tidak membantu, jadi epoch pendek saja tidak menggantikan regularisasi.

**Hal lain:**
- Label smoothing merugikan di kedua keputusan (argmax −3,69 poin, 0/5 fold). AUROC-nya turun ke 0,918 dan ECE naik ke 0,236. Dugaannya, label smoothing berbenturan dengan *logit-adjusted loss*.
- Tukar latar (R-latar) netral: argmax +1,22 poin, +bias_val −0,61 poin. Augmentasi ini belum terbukti mengurangi ketergantungan pada latar. Perlu diuji ulang dengan protokol H5d (mode lesi/latar), bukan hanya BACC.
- AUROC makro naik pada R-sd (0,9650), R-cutmix (0,9627), dan R-latar (0,9624) dibanding A0 (0,9575). Selisih ini tidak diuji.

**Keterbatasan:**
- 1×5 fold, dengan fold seed 0 yang sudah dipakai di 5 eksperimen.
- Hiperparameter memakai nilai bawaan paper, tanpa tuning.
- Temuan argmax ini muncul dari analisis sekunder, sehingga **wajib** diuji sebagai hipotesis baru yang ditetapkan sebelumnya pada split baru (seed 1 dan 2) sebelum diklaim.
