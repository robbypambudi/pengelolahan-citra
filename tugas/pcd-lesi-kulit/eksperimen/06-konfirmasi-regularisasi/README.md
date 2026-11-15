# Eksperimen 06: konfirmasi regularisasi + keputusan argmax pada split baru

Lanjutan dari eksperimen 05 (`../05-regularisasi/`, run `hasil/20261109-164154`).

Kode:
- `run.py`: memakai ulang `buat_model` dan `latih` dari `../05-regularisasi/run.py`, alur dari `../03-derm-v2/run.py`, lalu menambahkan blok uji `konfirmasi` ke `metrics.json`.
- Perubahan pada kode bersama: `../03-derm-v2/run.py` `main()` kini mengembalikan folder run. Perilaku lain tidak berubah.

## Latar belakang
Di eksperimen 05, hipotesis utama ("varian terpilih +bias_val > A0") tidak didukung. Namun analisis sekunder menemukan pola lain: dengan keputusan **argmax**, tiga varian unggul atas A0 di **5/5 fold** (split seed 0):

| Varian | BACC uji argmax | Δ vs A0 argmax | Δ vs A0 +bias_val (0,7522) |
|---|---|---|---|
| R-mixup | 0,7726 | +3,38 poin | +2,04 poin |
| R-sd | 0,7673 | +2,85 poin | +1,51 poin |
| R-cutmix | 0,7648 | +2,61 poin | +1,26 poin |

Setelah koreksi Holm atas 12 perbandingan, semuanya tidak bermakna (p_holm 0,35–0,77). Temuan ini muncul **setelah** melihat data, jadi belum boleh diklaim. Eksperimen ini mengujinya sebagai hipotesis yang ditetapkan sebelumnya, pada split yang belum pernah dipakai.

## Hipotesis (dibekukan SEBELUM run penuh, 15 Nov 2026)
- **H1 (primer):** untuk X ∈ {R-mixup, R-sd, R-cutmix}, BACC uji X|argmax > A0|argmax.
- **H2 (sekunder):** untuk X yang sama, BACC uji X|argmax > A0|+bias_val. Artinya regularisasi tanpa penyetelan bias mengalahkan *pipeline* A0 terbaik dari eksperimen 02–05.

## Aturan keputusan
1. **Data uji:** split StratifiedGroupKFold 5 fold per lesi dengan **seed ulangan 1 dan 2**, jadi 10 fold uji. Seed ini belum pernah dipakai di eksperimen 00–05. Hasil seed 0 **tidak digabung**, karena seed 0 adalah data yang memunculkan hipotesis.
2. **Varian dan hiperparameter** sama persis dengan eksperimen 05, tanpa tuning ulang:
   - mixup α = 0,2;
   - cutmix α = 1,0;
   - *stochastic depth* p maks 0,2.
3. **Uji statistik:** selisih BACC per fold (pasangan ulangan × fold yang sama) diuji dengan *corrected resampled t-test* (Nadeau & Bengio, 2003), memakai n = 10 dan rasio uji/latih dari run.
4. **Koreksi Holm** dilakukan per keluarga: H1 atas 3 perbandingan, H2 atas 3 perbandingan.
5. Sebuah perbandingan **didukung** bila selisih rata-rata ≥ SESOI 2 poin BACC **dan** p_holm < 0,05.
6. **H1 dianggap terkonfirmasi** bila minimal satu varian didukung. Varian yang tidak didukung dilaporkan apa adanya.
7. **R-cutmix-sd** (cutmix + *stochastic depth*) bersifat **eksploratif**. Varian ini di luar keluarga uji, dilaporkan tanpa klaim, dan hanya menjadi kandidat untuk eksperimen berikutnya.
8. Semua varian dan kedua keputusan dilaporkan, termasuk yang merugikan.

## Daya uji (dihitung sebelum run dari eksperimen 05)
- SD selisih per fold di eksperimen 05:
  - vs A0 argmax: 0,012–0,019;
  - vs A0 +bias_val: 0,011–0,014.
- Dengan rasio uji/latih 0,25, koreksi Nadeau–Bengio membuat galat baku hampir tidak mengecil walau jumlah fold ditambah: faktor √(1/k + 0,25) bernilai 0,59 pada k = 10 dan 0,56 pada k = 15.
- Perkiraan MDE (daya 80%, α Holm terketat 0,05/3) pada SD 0,010 / 0,015 / 0,020 adalah 2,3 / 3,4 / 4,5 poin.
- Efek di eksperimen 05 (+2,6 s.d. +3,4 poin untuk H1; +1,3 s.d. +2,0 poin untuk H2) berada di sekitar MDE. Artinya:
  - H1 punya peluang yang wajar untuk terdeteksi;
  - H2 kemungkinan besar tidak terdeteksi walaupun efeknya nyata.

  Hasil "tidak didukung" pada H2 **bukan** bukti bahwa efeknya tidak ada.

## Data
- Cache 256 px; validasi dalam 10% lesi.
- Split per lesi: seed ulangan 1 dan 2, n_fold 5.
- Pengecekan kebocoran `lesion_id` dilakukan otomatis.

## Varian
| Kode | Regime | Peran |
|---|---|---|
| A0 | – | acuan |
| R-mixup | mixup α = 0,2 | H1, H2 |
| R-sd | *stochastic depth* p maks 0,2 | H1, H2 |
| R-cutmix | cutmix α = 1,0 | H1, H2 |
| R-cutmix-sd | cutmix α = 1,0 + *stochastic depth* 0,2 | eksploratif |

## Keterbatasan yang diketahui sebelum run
- Split v0 belum memakai dedup pHash, sehingga duplikat lesi dengan `lesion_id` berbeda mungkin masih ada.
- Hasil ini juga belum berasal dari uji eksternal ISIC 2018. Konfirmasi di sini adalah konfirmasi **internal**.
- Seed pelatihan mengikuti rumus `42 + 1000·r + 10·f`, jadi seed latih juga baru.

## Cara menjalankan
```bash
uv run python tugas/pcd-lesi-kulit/eksperimen/06-konfirmasi-regularisasi/run.py --config config-cepat.yaml   # uji jalan
uv run python tugas/pcd-lesi-kulit/eksperimen/06-konfirmasi-regularisasi/run.py                               # penuh, ±1,7 jam
```

Uji jalan `hasil/20261115-150430` (1.500 citra, 2 epoch) hanya memastikan alur berjalan. Angkanya **tidak** untuk laporan.

## Hasil
Run `hasil/20261115-150601`:
- 5 varian × 10 fold, durasi 1,91 jam;
- dijalankan pada commit df38452 dengan *working tree* bersih;
- metrik tambahan ada di `metrik_tambahan.json` (selisih maks vs `metrics.json` 0,0007, akibat pembulatan bias).

**BACC uji (rata-rata ± SD, 10 fold):**

| Varian | argmax | +bias_val | AUROC makro (argmax) | ECE (argmax) | Sens. mel (argmax / +bias_val) |
|---|---|---|---|---|---|
| A0 | 0,7353 ± 0,0215 | 0,7444 ± 0,0211 | 0,9592 | 0,1100 | 50,0% / 62,6% |
| R-mixup | 0,7568 ± 0,0267 | 0,7420 ± 0,0330 | 0,9413 | 0,0729 | 53,2% / 60,6% |
| R-sd | 0,7532 ± 0,0206 | 0,7433 ± 0,0206 | 0,9643 | 0,0750 | 55,2% / 67,4% |
| **R-cutmix** | **0,7735 ± 0,0125** | 0,7615 ± 0,0156 | 0,9625 | **0,0221** | 55,3% / 58,5% |
| R-cutmix-sd | 0,7623 ± 0,0141 | 0,7482 ± 0,0195 | 0,9605 | 0,0300 | 54,8% / 65,6% |

**Uji konfirmasi** (*corrected resampled t-test*, Holm per keluarga, SESOI 2 poin):

| Perbandingan | Δ BACC | Δ ulangan 1 / 2 | Fold + | p | p_holm | Putusan |
|---|---|---|---|---|---|---|
| H1 R-mixup − A0 (argmax) | +2,15 | +2,67 / +1,63 | 9/10 | 0,047 | 0,094 | tidak didukung |
| H1 R-sd − A0 (argmax) | +1,79 | +1,49 / +2,10 | 8/10 | 0,120 | 0,120 | tidak didukung |
| H1 R-cutmix − A0 (argmax) | +3,82 | +3,05 / +4,58 | 9/10 | 0,019 | 0,056 | tidak didukung |
| H2 R-mixup argmax − A0 +bias_val | +1,24 | +2,53 / −0,04 | 8/10 | 0,337 | 0,675 | tidak didukung |
| H2 R-sd argmax − A0 +bias_val | +0,89 | +1,35 / +0,42 | 7/10 | 0,402 | 0,675 | tidak didukung |
| H2 R-cutmix argmax − A0 +bias_val | +2,91 | +2,92 / +2,91 | 9/10 | 0,040 | 0,120 | tidak didukung |
| eksploratif R-cutmix-sd − A0 (argmax) | +2,70 | +2,55 / +2,85 | 10/10 | 0,043 | – | tanpa klaim |
| eksploratif R-cutmix-sd argmax − A0 +bias_val | +1,79 | +2,41 / +1,18 | 8/10 | 0,171 | – | tanpa klaim |

**Metrik lain (argmax, rata-rata 10 fold):**

| Varian | F1 makro | Presisi makro | MCC | Brier | Log loss |
|---|---|---|---|---|---|
| A0 | **0,6906** | **0,6632** | **0,6708** | 0,2782 | 0,7497 |
| R-mixup | 0,6538 | 0,5989 | 0,6498 | 0,3072 | 0,6676 |
| R-sd | 0,6854 | 0,6454 | 0,6691 | **0,2656** | 0,5757 |
| R-cutmix | 0,6629 | 0,6095 | 0,6621 | 0,2700 | **0,5419** |
| R-cutmix-sd | 0,6317 | 0,5820 | 0,6399 | 0,2933 | 0,5909 |

Ringkasan bawaan alur eksperimen 03 memilih R-cutmix sebagai varian dengan BACC validasi tertinggi (0,7661). Ini sama dengan eksperimen 05.

## Kesimpulan
**Menurut aturan yang dibekukan, H1 dan H2 tidak didukung.** Tidak ada perbandingan yang memenuhi kedua syarat sekaligus (Δ ≥ 2 poin dan p_holm < 0,05).
- **R-cutmix** paling dekat:
  - Δ +3,82 poin, jauh di atas SESOI;
  - positif di 9/10 fold dan di kedua ulangan;
  - p mentah 0,019, tetapi p_holm 0,056, sedikit di atas α.
- **R-mixup:** Δ +2,15 poin melewati SESOI, tetapi p_holm 0,094.
- **R-sd:** Δ +1,79 poin, di bawah SESOI.

**Yang dapat dikatakan secara jujur:**
- **Arahnya konsisten.** Di dua split independen (seed 0 di eksperimen 05; seed 1 dan 2 di sini), CutMix dengan argmax selalu di atas A0 argmax:
  - +2,61 poin pada 5/5 fold;
  - +3,82 poin pada 9/10 fold.

  Ini bukti yang cukup kuat bahwa efeknya ada. Namun dengan protokol ini, efek tersebut **belum lolos uji yang ditetapkan sebelumnya**, jadi belum boleh ditulis sebagai "terbukti".
- Sesuai perhitungan daya uji sebelum run, koreksi Nadeau–Bengio dengan rasio uji/latih 0,25 membuat efek sekitar 3 poin berada di batas deteksi. Menambah fold hampir tidak membantu. Yang membantu adalah varians selisih yang lebih kecil, atau uji eksternal (ISIC 2018 Test) dengan bootstrap.
- Pola eksperimen 05 terulang: penyetelan bias (+bias_val) **menurunkan** BACC varian yang diregularisasi (R-cutmix −1,20 poin; R-mixup −1,48 poin), tetapi menaikkan A0 (+0,91 poin).

**Peringatan: BACC bukan satu-satunya sudut pandang.**
- Sensitivitas melanoma tertinggi tetap diraih +bias_val: R-sd 67,4%, R-cutmix-sd 65,6%, A0 62,6%. R-cutmix argmax hanya 55,3%. Bila tujuan klinisnya adalah tidak melewatkan melanoma, "cutmix + argmax" belum tentu pilihan terbaik.
- F1 makro, presisi makro, dan MCC tertinggi tetap dimiliki A0 argmax. Regularisasi menaikkan recall kelas minor (BACC) dengan mengorbankan presisi, sama seperti di eksperimen 05.
- Kalibrasi jelas lebih baik dengan CutMix: ECE 0,0221 vs A0 0,1100, dan log loss 0,5419 vs 0,7497. R-sd memiliki Brier terendah (0,2656). Selisih kalibrasi ini tidak diuji secara formal.
- R-mixup menurunkan AUROC makro (0,9413 vs 0,9592), padahal BACC-nya naik.

**Eksploratif:** CutMix + SD tidak lebih baik daripada CutMix saja (0,7623 vs 0,7735 argmax). Kombinasi ini tidak perlu dilanjutkan.

**Keterbatasan:**
- Kesimpulan ini masih bergantung pada split v0 tanpa dedup pHash.
- Hasilnya juga baru konfirmasi internal, belum diuji pada data eksternal.
- Hiperparameter tidak dituning, sehingga α CutMix atau Mixup yang lain bisa memberi hasil berbeda. Namun tuning harus dilakukan di validasi dalam, bukan di fold uji ini.

**Usulan langkah berikutnya:**
1. Bekukan **A0 + CutMix (argmax)** sebagai kandidat konfigurasi final, berdampingan dengan A0 +bias_val sebagai pembanding klinis (sensitivitas mel).
2. Uji keduanya pada ISIC 2018 Test (1.512 citra) dengan bootstrap CI. Ini adalah uji yang ditunda di `riset/PLAN.md`.
3. Sebelum itu, buat split v1 dengan dedup pHash agar uji eksternal bersih dari duplikat.
