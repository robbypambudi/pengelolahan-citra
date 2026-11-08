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
| (run penuh) | penuh | `[BELUM DIJALANKAN]` | |

## Kesimpulan
`[BELUM DIJALANKAN]`
