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
[BELUM DIJALANKAN]

## Kesimpulan
[BELUM DIJALANKAN]
