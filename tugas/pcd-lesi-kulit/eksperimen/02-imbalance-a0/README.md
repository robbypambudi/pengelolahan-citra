# Eksperimen 02: Strategi ketidakseimbangan kelas pada A0

Lanjutan dari eksperimen 01 (`../01-derm-mobilenetv3/`, run `hasil/20261108-222958`). Kode bersama: `citra/imbalance.py` (tes: `tests/test_imbalance.py`).

## Pertanyaan
Pada MobileNetV3-Large polos (A0), dengan protokol per lesi yang sama dengan eksperimen 01, strategi penanganan ketidakseimbangan kelas mana yang memberi BACC terbaik? Apakah ada yang mengurangi kebingungan melanoma ↔ nevus?

Latar belakang dari eksperimen 01 (A0, 5 fold dijumlah):
- *Recall* melanoma 0,54; 318 melanoma tertebak nevus dan 350 nevus tertebak melanoma.
- AUROC makro 0,96 sedangkan BACC 0,73. Ada dugaan ruang perbaikan pada aturan keputusan.

## Aturan pemilihan (ditetapkan SEBELUM run penuh, 9 Nov 2026)
1. **Pemenang = strategi dengan rata-rata BACC validasi dalam tertinggi** (epoch akhir, keputusan argmax). Fold uji tidak dipakai untuk memilih.
2. Pemenang dipakai untuk semua varian di eksperimen berikutnya (03, Derm-MobileNetV3 v2).
3. Bias keputusan per kelas (`bias_val`) dicari hanya di validasi dalam, lalu diterapkan ke fold uji. Ini dilaporkan terpisah sebagai "+bias_val".
4. Semua strategi dan semua keputusan dilaporkan, termasuk yang kalah.

## Data
Sama dengan eksperimen 01:
- cache `data/olahan/ham10000-256`
- StratifiedGroupKFold 5 fold per lesi dengan seed 0 (fold identik dengan eksperimen 01)
- validasi dalam 10% lesi
- seed pelatihan sama per fold

## Metode
Model A0, 20 epoch, model epoch terakhir; semua hiperparameter sama dengan eksperimen 01.

| Kode | Loss | Bobot kelas | Sampler | Keputusan tambahan |
|---|---|---|---|---|
| S1-bobot-penuh | CE | 1/n_k (= eksperimen 01, **acuan**) | acak | – |
| S0-tanpa | CE | – | acak | koreksi prior τ=1 (Menon dkk., 2021) |
| S2-bobot-akar | CE | 1/√n_k | acak | – |
| S3-bobot-efektif | CE | (1−β)/(1−β^n_k), β=0,999 (Cui dkk., 2019) | acak | – |
| S4-sampler | CE | – | seimbang (peluang ∝ 1/n_k, hanya data latih) | – |
| S5-focal-akar | Focal γ=2 (Lin dkk., 2017) | 1/√n_k | acak | – |
| S6-logit-adj | CE pada logit + log π (Menon dkk., 2021) | – | acak | – |

Semua strategi juga dievaluasi dengan "+bias_val", yaitu bias log-probabilitas per kelas yang memaksimalkan BACC validasi dalam (*coordinate ascent*, grid −3…3, langkah 0,1).

## Metrik
- BACC (utama), sensitivitas melanoma, sens. mel @ spes 80%, F1 makro, AUROC makro, ECE, dan *confusion matrix* per fold.
- Uji vs S1|argmax memakai *corrected resampled t-test* + Holm.

## Cara menjalankan
```bash
uv run python tugas/pcd-lesi-kulit/eksperimen/02-imbalance-a0/run.py --config config-cepat.yaml   # uji jalan
uv run python tugas/pcd-lesi-kulit/eksperimen/02-imbalance-a0/run.py                               # penuh, ±1,6 jam
```

## Hasil
| Run | Mode | Pemenang (validasi) | Catatan |
|---|---|---|---|
| `hasil/20261109-010731` | cepat (1 fold, 2 epoch, 1.500 citra) | – | Uji jalan; angkanya **tidak** untuk laporan. S1 = 0,3478, identik dengan A0 di uji jalan eksperimen 01 (cek konsistensi). |
| `hasil/20261109-010819` | penuh, 1×5 fold, 20 epoch | **S6-logit-adj** | Run acuan. Commit `0273ded`, tanpa perubahan belum di-commit. 1,21 jam. |

### Run `hasil/20261109-010819` (5 fold uji, rata-rata ± SD)
Uji vs S1|argmax memakai *corrected resampled t-test* (rasio uji/latih 0,25), koreksi Holm atas 14 perbandingan.

| Strategi | Keputusan | BACC val | BACC uji | Sens. mel | F1 makro | ECE | Δ vs S1 | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| S1-bobot-penuh | argmax | 0,7160 | 0,7301 ± 0,0263 | 0,539 | 0,7146 | 0,089 | – | – |
| S1-bobot-penuh | +bias_val | 0,7834 | 0,7523 ± 0,0216 | 0,593 | 0,6596 | 0,128 | +0,0222 | 1,000 |
| S0-tanpa | argmax | 0,6898 | 0,6888 ± 0,0188 | 0,464 | 0,7135 | 0,109 | −0,0412 | 0,233 |
| S0-tanpa | +prior | 0,7330 | 0,7378 ± 0,0297 | 0,511 | 0,7031 | 0,111 | +0,0077 | 1,000 |
| S0-tanpa | +bias_val | 0,7698 | 0,7060 ± 0,0397 | 0,615 | 0,6782 | 0,129 | −0,0241 | 1,000 |
| S2-bobot-akar | argmax | 0,6976 | 0,7144 ± 0,0225 | 0,483 | 0,7273 | 0,103 | −0,0156 | 0,314 |
| S2-bobot-akar | +bias_val | 0,7903 | 0,7431 ± 0,0372 | 0,659 | 0,6924 | 0,129 | +0,0130 | 1,000 |
| S3-bobot-efektif | argmax | 0,6912 | 0,7148 ± 0,0207 | 0,463 | 0,7194 | 0,103 | −0,0152 | 1,000 |
| S3-bobot-efektif | +bias_val | 0,7701 | 0,7238 ± 0,0344 | 0,632 | 0,6926 | 0,125 | −0,0063 | 1,000 |
| S4-sampler | argmax | 0,6801 | 0,7032 ± 0,0226 | 0,529 | 0,7145 | 0,104 | −0,0269 | 0,157 |
| S4-sampler | +bias_val | 0,7811 | 0,7328 ± 0,0386 | 0,717 | 0,6758 | 0,144 | +0,0027 | 1,000 |
| S5-focal-akar | argmax | 0,6911 | 0,7072 ± 0,0203 | 0,482 | 0,7200 | **0,027** | −0,0228 | 1,000 |
| S5-focal-akar | +bias_val | 0,7903 | 0,7217 ± 0,0586 | 0,690 | 0,6495 | 0,057 | −0,0083 | 1,000 |
| **S6-logit-adj** | argmax | **0,7407** | 0,7388 ± 0,0212 | 0,500 | 0,6945 | 0,108 | +0,0087 | 1,000 |
| **S6-logit-adj** | +bias_val | 0,8031 | **0,7522 ± 0,0230** | 0,606 | 0,6613 | 0,120 | **+0,0221** (5/5 fold positif) | **0,022** |

Cek konsistensi: S1|argmax = 0,7301 ± 0,0263, sama persis dengan A0 di eksperimen 01 (run 20261108-222958). Pipeline terbukti deterministik.

*Confusion* melanoma ↔ nevus (5 fold dijumlah, argmax):
- S1: 318 mel → nv, 350 nv → mel, *recall* mel 0,54
- S6: 351 mel → nv, 237 nv → mel, *recall* mel 0,50


## Kesimpulan
- **Pemenang menurut aturan yang ditetapkan sebelum run: S6-logit-adj** (BACC validasi 0,7407). Di fold uji, keunggulannya kecil dan belum bermakna: +0,87 poin vs S1 (p_holm 1,000).
- **Bias keputusan dari validasi dalam (+bias_val) menambah sekitar 2 poin BACC.** Pada S6 kenaikannya konsisten di kelima fold: +2,21 poin, p_holm 0,022, dan melewati SESOI 2 poin. Pada S1 kenaikannya sama besar tetapi lebih bervariasi (p_holm 1,000).
- Harga +bias_val: F1 makro turun (0,69 → 0,66) dan kalibrasi memburuk (ECE 0,108 → 0,120). Bias ini memindahkan tebakan ke kelas kecil, jadi presisinya turun.
- **Kebingungan melanoma ↔ nevus tidak teratasi oleh strategi mana pun.** *Recall* melanoma (argmax) tetap 0,46–0,54. Ini mendukung dugaan bahwa masalahnya kemiripan visual, bukan jumlah data.
- Strategi yang lebih "lunak" (akar, efektif, focal) dan sampler seimbang **menurunkan** BACC argmax 1,5–2,7 poin vs bobot penuh. Tanpa penanganan sama sekali (S0) turun 4,1 poin.
- Temuan sampingan: *focal loss* memberi kalibrasi jauh lebih baik (ECE 0,027 vs ±0,10).

**Keputusan untuk eksperimen berikutnya:** S6-logit-adj + bias_val (keduanya dari data latih/validasi dalam) dipakai untuk semua varian.

Keterbatasan: 1×5 fold; bias dicari di validasi kecil (±800 citra, kelas kecil ±10 citra), sehingga bisa *overfit* (BACC val +bias 0,80 vs uji 0,75).

