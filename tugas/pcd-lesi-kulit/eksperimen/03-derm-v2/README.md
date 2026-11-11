# Eksperimen 03: Derm-MobileNetV3 v2 (letak M3, lebar cincin) dan uji jalan pintas H5d

Lanjutan dari:
- eksperimen 01 (`../01-derm-mobilenetv3/`, run `hasil/20261108-222958`)
- eksperimen 02 (`../02-imbalance-a0/`, run `hasil/20261109-010819`)

Kode bersama: `citra/derm.py`, termasuk `TutupInput` untuk H5d, dengan tes di `tests/test_derm.py`.

## Pertanyaan
1. **Letak dan lebar M3.** Di eksperimen 01, M3 di blok 16 (7×7, cincin 16 px ≈ 0,5 sel peta fitur) tidak menaikkan BACC (A3 0,7253 vs A0 0,7301). Apakah M3 berguna bila diletakkan di peta fitur yang lebih halus (blok 12: 14×14, blok 6: 28×28) dan cincinnya lebih lebar?
2. **H5d (uji jalan pintas, desain diperbaiki).** Di eksperimen 01, A3-L/A3-C tetap memakai GAP seluruh citra, jadi H5d tidak teruji. Di sini penutupan dilakukan di **input**. Model hanya melihat lesi (A0-lesi) atau hanya kulit di sekitar dan latar (A0-latar). Seberapa jauh A0-latar di bawah A0-lesi?

## Aturan (ditetapkan SEBELUM run penuh, 9 Nov 2026)
1. **Strategi latih untuk semua varian:** S6-logit-adj, pemenang eksperimen 02. Dua keputusan dilaporkan: argmax dan +bias_val (bias per kelas dicari di validasi dalam). **Keputusan utama: +bias_val**, sesuai kesimpulan eksperimen 02.
2. **Varian A3 terpilih** = BACC validasi dalam rata-rata tertinggi (argmax) di antara varian A3. BACC +bias_val di validasi tidak dipakai, karena bias dicari di data yang sama. Fold uji tidak dipakai untuk memilih.
3. **H5a** (M3 > A0) didukung hanya bila varian A3 terpilih unggul ≥ SESOI 2 poin BACC (+bias_val) dan p_holm < 0,05. Koreksi Holm diterapkan atas 10 perbandingan, yaitu 5 varian A3 × 2 keputusan.
4. **H5d** didukung bila A0-lesi − A0-latar ≥ 10 poin BACC (+bias_val) dan p_holm < 0,05. Ambang 10 poin = 5 × SESOI; ini asumsi kami. Bila A0-latar jauh di atas peluang (BACC 1/7 ≈ 0,143) dan dekat A0, berarti model bisa menebak dari latar. Dalam kasus itu, klaim H5a harus dibatasi.
5. Semua varian dilaporkan, termasuk yang kalah. A0 harus sama persis dengan S6 eksperimen 02 (cek konsistensi).

## Data
Sama dengan eksperimen 01/02:
- cache `data/olahan/ham10000-256`
- StratifiedGroupKFold 5 fold per lesi, seed 0
- validasi dalam 10% lesi
- seed pelatihan sama per fold

Mask lesi: Otsu. Pada 439 dari 10.015 citra, Otsu gagal dan mask diganti elips di tengah. BACC uji juga dilaporkan pada subset Otsu-berhasil (`bacc_uji_otsu_ok`), karena H5d sangat bergantung pada mutu mask.

## Metode
Semua hiperparameter sama dengan eksperimen 02 (20 epoch, epoch terakhir). Hanya varian yang berubah.

| Kode | Isi | Blok M3 (kanal, ukuran) | Cincin | Untuk |
|---|---|---|---|---|
| A0 | MobileNetV3-Large polos | – | – | acuan |
| A3-b16-r16 | + M3 penuh (= A3 eksperimen 01) | 16 (960, 7×7) | 16 px | H5a |
| A3-b12-r16 | + M3 penuh | 12 (112, 14×14) | 16 px | H5a, letak |
| A3-b12-r32 | + M3 penuh | 12 | 32 px | H5a, lebar |
| A3-b12-r48 | + M3 penuh | 12 | 48 px | H5a, lebar |
| A3-b6-r16 | + M3 penuh | 6 (40, 28×28) | 16 px | H5a, letak |
| A0-lesi | A0, piksel di luar mask diganti warna rata-rata ImageNet | – | – | H5d |
| A0-latar | A0, lesi + 16 px tepinya diganti warna rata-rata | – | – | H5d |

Penutupan dilakukan setelah augmentasi, dengan mask yang ikut diaugmentasi, saat latih maupun uji. Contoh input ada di `hasil/<run>/contoh/`.

## Metrik
- Sama dengan eksperimen 02: BACC (utama), sensitivitas melanoma, sens. mel @ spes 80%, F1 makro, AUROC makro, ECE, dan *confusion matrix*.
- Tambahan: BACC uji pada subset Otsu-berhasil, serta parameter tambahan dan FLOPs per varian.
- Uji statistik memakai *corrected resampled t-test* + Holm, dengan dua keluarga uji terpisah: A3 vs A0, dan H5d.

## Keterbatasan yang diketahui sebelum run
- **Mask Otsu cenderung sempit.** Halo pigmen tipis di tepi lesi sering tidak masuk mask (lihat contoh di uji jalan). Akibatnya:
  - A0-lesi kehilangan sebagian tepi lesi.
  - Pada A0-latar, dilasi 8 px masih membocorkan halo, jadi dinaikkan ke 16 px.
  - Cincin M3 (mulai tepat di batas mask) ikut berisi halo itu.
- Siluet mask terlihat di A0-lesi maupun A0-latar, sehingga bentuk lesi tetap bocor di keduanya.
- 1×5 fold.

## Cara menjalankan
```bash
uv run python tugas/pcd-lesi-kulit/eksperimen/03-derm-v2/run.py --config config-cepat.yaml   # uji jalan
uv run python tugas/pcd-lesi-kulit/eksperimen/03-derm-v2/run.py                               # penuh, ±2 jam
uv run python tugas/pcd-lesi-kulit/eksperimen/03-derm-v2/run.py --lanjut hasil/<waktu>        # lanjutkan run terputus
```

## Hasil
| Run | Mode | Catatan |
|---|---|---|
| `hasil/20261109-054345` | cepat (1 fold, 2 epoch, 1.500 citra) | Uji jalan; angkanya **tidak** untuk laporan. A0 = S6 uji jalan eksperimen 02, identik (BACC 0,2029 argmax, 0,3446 +bias_val). Dilasi masih 8 px. |
| `hasil/20261109-054525` | penuh, 1×5 fold, 20 epoch | Run acuan. 1,61 jam. Basis commit `9eca953` + perubahan belum di-commit (disimpan di `perubahan_belum_commit.patch`). |

### Run `hasil/20261109-054525` (5 fold uji, rata-rata ± SD)
Cek konsistensi: A0 = 0,7388 (argmax) / 0,7522 (+bias_val). Hasil ini sama persis dengan S6 di eksperimen 02.

| Varian | Param. tambahan | BACC val (argmax) | BACC uji argmax | BACC uji +bias_val | Sens. mel (+bias) | AUROC makro (argmax) | F1 makro (+bias) |
|---|---|---|---|---|---|---|---|
| A0 | 0 | 0,7407 | 0,7388 ± 0,0212 | **0,7522 ± 0,0230** | 0,606 | 0,9575 | 0,6613 |
| A3-b16-r16 | 498.369 | 0,7166 | 0,7242 ± 0,0186 | 0,7468 ± 0,0204 | 0,618 | 0,9611 | 0,6376 |
| A3-b12-r16 | 167.649 | 0,7145 | 0,7189 ± 0,0140 | 0,7303 ± 0,0260 | 0,649 | 0,9611 | 0,6387 |
| **A3-b12-r32** (terpilih) | 167.649 | **0,7328** | 0,7255 ± 0,0152 | 0,7425 ± 0,0184 | 0,536 | 0,9601 | 0,6581 |
| A3-b12-r48 | 167.649 | 0,7228 | 0,7285 ± 0,0132 | 0,7324 ± 0,0243 | 0,609 | 0,9602 | 0,6363 |
| A3-b6-r16 | 139.569 | 0,7250 | 0,7360 ± 0,0313 | 0,7506 ± 0,0226 | 0,611 | 0,9626 | 0,6409 |
| A0-lesi | 0 | 0,6074 | 0,6300 ± 0,0294 | 0,6262 ± 0,0216 | 0,537 | 0,9252 | 0,4904 |
| A0-latar | 0 | 0,4793 | 0,4794 ± 0,0253 | 0,4846 ± 0,0271 | 0,455 | 0,8686 | 0,3626 |

FLOPs semua varian ≈ 431 juta per citra. Tambahan M3 paling besar ≈ 1 juta FLOPs.

**A3 vs A0** (selisih BACC uji, *corrected resampled t-test*, Holm atas 10 perbandingan):

| Varian | argmax | +bias_val | Fold positif (+bias) | p_holm (+bias) |
|---|---|---|---|---|
| A3-b16-r16 | −0,0146 | −0,0054 | 2/5 | 1,000 |
| A3-b12-r16 | −0,0199 | −0,0219 | 0/5 | 0,750 |
| A3-b12-r32 | −0,0133 | −0,0097 | 1/5 | 1,000 |
| A3-b12-r48 | −0,0103 | −0,0199 | 1/5 | 1,000 |
| A3-b6-r16 | −0,0028 | −0,0016 | 3/5 | 1,000 |

**H5d** (Holm atas 6 perbandingan):

| Perbandingan | argmax | +bias_val | Fold positif | p_holm (+bias) |
|---|---|---|---|---|
| A0-lesi − A0-latar | +0,1506 | **+0,1417** | 5/5 | 0,013 |
| A0 − A0-lesi | +0,1088 | +0,1260 | 5/5 | 0,010 |
| A0 − A0-latar | +0,2594 | +0,2676 | 5/5 | 0,001 |

Subset Otsu-berhasil (argmax) memberi pola yang sama: A0 0,7382, A0-lesi 0,6252, A0-latar 0,4842.

*Recall* per kelas (argmax, 5 fold dijumlah; urutan akiec, bcc, bkl, df, mel, nv, vasc):

| Varian | Recall |
|---|---|
| A0 | 0,70 / 0,76 / 0,69 / 0,72 / 0,50 / 0,92 / 0,88 |
| A0-lesi | 0,57 / 0,64 / 0,57 / 0,49 / 0,48 / 0,82 / 0,85 |
| A0-latar | 0,57 / 0,42 / 0,43 / 0,36 / 0,33 / 0,83 / 0,42 |

Gerbang M3 di akhir latih: 0,20–0,32 (blok 16) dan 0,36–0,46 (blok 12/6). Norma bobot akhir cabang ≈ 10–17. Jadi cabang M3 aktif dan dipakai model, bukan diam di nol.

## Kesimpulan
**H5a tidak didukung.**
- Varian A3 terpilih menurut aturan (A3-b12-r32, BACC validasi 0,7328) justru 0,97 poin di bawah A0 (+bias_val), hanya positif di 1 dari 5 fold, p_holm 1,000.
- Tidak satu pun dari lima konfigurasi M3 mengalahkan A0. Yang paling dekat adalah A3-b6-r16 (−0,16 poin, 3/5 fold).
- Memindah M3 ke peta fitur yang lebih halus dan melebarkan cincin tidak mengubah kesimpulan eksperimen 01.
- Cabang M3 aktif (gerbang 0,2–0,46) dan AUROC makro sedikit lebih tinggi (0,960–0,963 vs 0,958; tidak diuji). Namun informasi tambahan itu tidak berubah menjadi BACC yang lebih baik.

**H5d didukung menurut kriteria yang ditetapkan sebelum run, tetapi dengan catatan penting.**
- A0-lesi − A0-latar = +14,2 poin (+bias_val), 5/5 fold, p_holm 0,013. Ini melewati ambang 10 poin.
- Namun A0-latar sendiri mencapai BACC 0,485, yaitu 3,4 kali tingkat peluang (0,143), padahal lesinya ditutup. Kulit sekitar, latar, dan siluet mask ternyata cukup informatif, terutama untuk nevus (*recall* 0,83) dan akiec (0,57). Ini sejalan dengan temuan Bissoto dkk. (2019).
- **A0 utuh juga jauh di atas A0-lesi** (+12,6 poin). Model memanfaatkan informasi di luar mask Otsu. Penyebabnya bisa (a) tepi lesi yang tidak masuk mask Otsu yang sempit, atau (b) jalan pintas dari latar. Desain ini belum bisa memisahkan keduanya.

**Implikasi untuk RQ5:**
- Dalam dua eksperimen (6 konfigurasi M3), M3 tidak memberi keuntungan. Klaim "pooling lesi–sekitar menaikkan kinerja" tidak didukung data kita.
- Temuan yang lebih kuat justru dari H5d: kinerja MobileNetV3 pada HAM10000 sebagian besar bergantung pada informasi di luar lesi tersegmentasi.

Keterbatasan:
- 1×5 fold.
- Mask Otsu sempit, dan 439 citra memakai elips cadangan.
- Siluet mask bocor di A0-lesi dan A0-latar.
- Ambang H5d 10 poin adalah asumsi kami.
