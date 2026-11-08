# Eksperimen 01: Ablasi Derm-MobileNetV3 (RQ5)

Bagian dari RQ5 di `riset/PROPOSAL.md` (v1.1). Desain dan posisi literatur: `riset/LITERATUR-DERM-MOBILENETV3.md`. Kode model: `citra/derm.py`. Segmentasi: `citra/segmentasi.py`. Split: `citra/split.py`.

## Pertanyaan / hipotesis
Apakah menambahkan operasi pengolahan citra yang bisa dilatih ke MobileNetV3-Large menaikkan BACC dan sensitivitas melanoma, bila dievaluasi dengan split per lesi?
- **H5a.** A3 (+M3) > A0 dengan selisih ≥ 2 poin BACC (SESOI).
- **H5b.** A1 (+M1 terlatih) tidak lebih buruk dari A0+P (operator M1 yang sama, tetapi tetap).
- **H5c.** A4 (M1 + M3) > AL (modifikasi blok ala Lilhore) dengan selisih ≥ SESOI.
- **H5d.** A3-C (cincin saja) jauh di bawah A3-L (lesi saja). Bila tidak, ada indikasi jalan pintas lewat latar (Bissoto dkk., 2019).

## Data
- HAM10000 = ISIC 2018 Task 3 Training: 10.015 citra, 7.470 lesi unik. Lokasi `data/mentah/isic2018/`, lisensi CC BY-NC 4.0.
- *Cache* (`siapkan.py`) di `data/olahan/ham10000-256/`:
  - citra di-*resize* ke 256×256
  - mask Otsu; 439 citra (4,4%) jatuh ke elips cadangan
- Split: StratifiedGroupKFold 5 fold, grup = `lesion_id`.
  - Fold disimpan di `data/split/folds-v0-lesion.csv` (dibuat saat run penuh).
  - Tes `tests/test_split.py` memastikan tidak ada lesi yang muncul di dua fold.
  - Validasi dalam = 10% lesi dari data latih tiap fold. Hanya dipakai untuk memilih epoch terbaik.
- **Belum**: penggabungan duplikat dengan *perceptual hash* (Plan 1) dan uji hold-out ISIC 2018 Test 1.512 citra.

## Metode
| Kode | Konfigurasi | Hipotesis |
|---|---|---|
| A0 | MobileNetV3-Large pralatih ImageNet (baseline) | acuan |
| A0+P | A0 + operator M1 dengan gerbang = 1 dan parameter dibekukan (SoG p=6 + pelemah rambut) | pembanding H5b |
| AL | A0 dengan SE → ECA + *dilated conv* tahap akhir (pendekatan Lilhore) | H5c |
| A1 | A0 + M1 terlatih | H5b |
| A3 | A0 + M3 (lesi, cincin, selisih) | H5a |
| A3-L / A3-C | M3 hanya fitur lesi / hanya fitur cincin | H5d |
| A4 | A0 + M1 + M3 | H5c |

Semua varian memakai hiperparameter, augmentasi, seed, dan split yang sama (lihat `config.yaml`).

Biaya model diukur dengan `hitung_biaya`; FLOPs menghitung 1 MAC = 2 FLOP.

| Varian | Parameter | Tambahan |
|---|---|---|
| A0 | 4.210.999 | – |
| A4 | 4.709.373 | +498.374 |
| AL | 2.698.753 | – |

## Asumsi (belum diatur proposal)
| Kode | Asumsi | Alasan |
|---|---|---|
| A1 | Hiperparameter tetap untuk semua varian (LR 3e-4 backbone, 3e-3 modul, 20 epoch, cosine + warmup 1 epoch, AdamW) | Anggaran tuning sama (nol). Bayesian optimization (proposal) menyusul bila waktu cukup. |
| A2 | Resize langsung 600×450 → 256 → 224, rasio aspek tidak dipertahankan | Sederhana; sama untuk semua varian |
| A3 | Augmentasi: rotasi 0–360°, flip, zoom 0,8–1,0, geser 10%, kecerahan/kontras ±10%; tanpa jitter hue | Lesi dermoskopi tidak punya orientasi baku. Jitter warna dihindari karena akan mengacaukan uji *color constancy*. |
| A4 | Ketidakseimbangan kelas ditangani dengan bobot loss n/(K·n_k), tanpa *oversampling* | Menghindari salinan citra yang sama dalam satu batch |
| A5 | Model yang dipakai = epoch terakhir (20) di akhir jadwal cosine; BACC validasi dalam hanya dipantau | Pada run 20261108-220753, BACC validasi naik-turun ±0,07 antar-epoch (kelas kecil hanya ±10 citra di validasi), sehingga memilih epoch "terbaik" lebih banyak menangkap derau. Fold uji tidak pernah dipakai untuk memilih model. |
| A6 | Cincin selebar 16 px pada 224; M3 di blok 16 (7×7) | Pilihan awal; lebar cincin dan blok bisa diablasi |
| A7 | Mixed precision bf16; M1 tetap float32 | Kecepatan; `pow` dengan p besar tidak stabil di bf16 |
| A8 | AL tanpa *cross-layer connection*, versi "lebih dalam/lebar", dan *bias loss* | Tidak dirinci di Lilhore dkk. (2024); dicatat di tabel kesetiaan |

## Metrik
- Utama: **BACC** (*balanced accuracy*).
- Sekunder: sensitivitas melanoma, sensitivitas melanoma @ spesifisitas 80%, F1 makro, AUROC makro, ECE (15 bin), *recall* per kelas, *confusion matrix*.
- Statistik vs A0: *corrected resampled t-test* (Nadeau & Bengio, 2003) atas selisih BACC per fold, koreksi Holm. "Lolos SESOI" = selisih ≥ 0,02 dan p_holm < 0,05.
- Dicatat juga: parameter M1/M3 yang dipelajari (p, gerbang, a, t) per epoch, dan prediksi per citra di `preds.csv`.

## Cara menjalankan
```bash
uv sync --extra dl
uv run python tugas/pcd-lesi-kulit/eksperimen/01-derm-mobilenetv3/siapkan.py              # sekali, ±15 detik
uv run python tugas/pcd-lesi-kulit/eksperimen/01-derm-mobilenetv3/run.py --config config-cepat.yaml   # uji jalan
uv run python tugas/pcd-lesi-kulit/eksperimen/01-derm-mobilenetv3/run.py                  # penuh: 8 varian × 5 fold
uv run python tugas/pcd-lesi-kulit/eksperimen/01-derm-mobilenetv3/run.py --lanjut hasil/<waktu>      # lanjutkan bila terputus
```

## Hasil
| Run | Mode | BACC A0 | BACC A4 | Catatan |
|---|---|---|---|---|
| `hasil/20261108-215858` | cepat (1 fold, 2 epoch, 1.500 citra) | – | – | Uji jalan saja; angkanya **tidak** untuk laporan |
| `hasil/20261108-220753` | penuh, **dihentikan** setelah fold 0 | – | – | Desain lama (gerbang M3/M2 = 0, early stopping). Diganti; angkanya **tidak** untuk laporan. Lihat catatan versi di `citra/derm.py`. |
| `hasil/20261108-222958` | penuh, 1×5 fold, 20 epoch | 0,7301 ± 0,0263 | 0,7117 ± 0,0255 | Run acuan. Commit `db76551` + perubahan belum di-commit (`citra/derm.py` v2, `run.py`). 1,59 jam. |

### Run `hasil/20261108-222958` (5 fold uji, rata-rata ± SD antar-fold)
Semua angka dari `metrics.json` run ini. Uji vs A0 memakai *corrected resampled t-test* (rasio uji/latih 0,25), koreksi Holm atas 7 perbandingan.

| Varian | BACC | Sens. melanoma | Sens. mel @ spes 80% | F1 makro | AUROC makro | ECE | Δ BACC vs A0 | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| A0 | 0,7301 ± 0,0263 | 0,539 ± 0,059 | 0,832 ± 0,029 | 0,7146 ± 0,0261 | 0,9604 ± 0,0040 | 0,0888 | – | – |
| A0+P | 0,6949 ± 0,0066 | 0,530 ± 0,019 | 0,801 ± 0,032 | 0,6772 ± 0,0111 | 0,9536 ± 0,0054 | 0,0896 | −0,0352 | 0,549 |
| AL | 0,7263 ± 0,0368 | 0,613 ± 0,057 | 0,844 ± 0,058 | 0,6821 ± 0,0326 | 0,9588 ± 0,0047 | 0,0731 | −0,0038 | 1,000 |
| A1 | 0,7275 ± 0,0166 | 0,542 ± 0,047 | 0,846 ± 0,027 | 0,7155 ± 0,0167 | 0,9602 ± 0,0052 | 0,0888 | −0,0025 | 1,000 |
| A3 | 0,7253 ± 0,0159 | 0,548 ± 0,036 | 0,808 ± 0,031 | 0,7120 ± 0,0198 | 0,9586 ± 0,0028 | 0,0918 | −0,0048 | 1,000 |
| A3-L | 0,7237 ± 0,0171 | 0,553 ± 0,070 | 0,844 ± 0,040 | 0,7104 ± 0,0235 | 0,9612 ± 0,0034 | 0,0876 | −0,0064 | 1,000 |
| A3-C | 0,7219 ± 0,0210 | 0,526 ± 0,034 | 0,809 ± 0,026 | 0,7045 ± 0,0174 | 0,9581 ± 0,0035 | 0,0947 | −0,0082 | 1,000 |
| A4 | 0,7117 ± 0,0255 | 0,544 ± 0,034 | 0,807 ± 0,034 | 0,6969 ± 0,0228 | 0,9582 ± 0,0040 | 0,0879 | −0,0184 | 0,104 |

Perbandingan berpasangan lain (uji t terkoreksi, tanpa Holm; dihitung dari `metrics.json` yang sama):
- A1 − A0+P: +0,0327 (lima fold semuanya positif), p = 0,056
- A4 − AL: −0,0146, p = 0,453
- A3-L − A3-C: +0,0018, p = 0,845

Parameter yang dipelajari (5 fold):
- A1: gerbang CC 0,19–0,25, p 5,59–5,91; gerbang rambut **negatif** (−0,08 s/d −0,21), artinya struktur gelap tipis justru diperkuat, bukan dilemahkan.
- A3: gerbang M3 0,21–0,27.

Biaya: A3/A4 +0,50 juta parameter (4,71 juta vs 4,21 juta), FLOPs praktis sama. AL 2,70 juta parameter tetapi 0,704 GFLOPs (+63%).

## Kesimpulan
Run acuan `hasil/20261108-222958`, 1×5 fold:
- **H5a tidak didukung.** A3 tidak lebih baik dari A0 (Δ −0,0048; p_holm 1,000).
- **H5b didukung secara arah.** M1 yang dilatih (A1) tidak lebih buruk dari praproses tetap (A0+P), malah 3,3 poin lebih tinggi di kelima fold (p = 0,056). Uji *non-inferiority* formal (TOST) belum dijalankan. Temuan sampingan: praproses tetap pada kekuatan penuh (A0+P) **menurunkan** BACC 3,5 poin vs A0 (tidak signifikan setelah Holm), dan model yang dilatih memilih memakai CC hanya ±20%.
- **H5c tidak didukung.** A4 < AL (Δ −0,0146; p = 0,453). A4 bahkan lebih rendah dari A0 di kelima fold (Δ −0,0184; p mentah 0,015; p_holm 0,104), tetapi selisihnya di bawah SESOI.
- **H5d belum bisa diuji dengan desain ini.** Di implementasi sekarang, A3-L dan A3-C tetap menjumlahkan GAP seluruh citra, jadi "lesi saja" dan "cincin saja" tidak benar-benar memisahkan informasi. Hasilnya hampir sama (Δ +0,0018). Uji jalan pintas yang benar perlu varian tanpa GAP global, atau citra yang ditutup (*masking*) di input.
- Semua varian punya AUROC makro 0,958–0,961, jadi kemampuan membedakan kelas hampir identik. Perbedaan BACC lebih banyak berasal dari ambang keputusan per kelas.

Keterbatasan:
- Hanya 5 fold, sehingga daya uji rendah. SD antar-fold 0,016–0,037, sehingga selisih di bawah ±3 poin sulit dibedakan dari derau.
- Hiperparameter tidak di-*tuning* per varian.
- Split belum memakai dedup pHash.
- Belum ada uji hold-out eksternal.
- Mask Otsu kasar pada lesi berkontras rendah (pembanding U-Net belum ada).
