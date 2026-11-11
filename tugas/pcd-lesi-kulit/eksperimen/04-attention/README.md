# Eksperimen 04: MobileNetV3-Large + CBAM / Coordinate Attention

Lanjutan dari eksperimen 03 (`../03-derm-v2/`, run `hasil/20261109-054525`).

Kode:
- `citra/attention.py`, dengan tes di `tests/test_attention.py`
- alur latih/evaluasi memakai ulang `../03-derm-v2/run.py`

## Pertanyaan
Apakah menambah modul attention ke MobileNetV3-Large menaikkan BACC dibanding A0, pada protokol per lesi yang sama? Modul yang diuji:
- CBAM (Woo dkk., 2018)
- Coordinate Attention / CA (Hou dkk., 2021)

## Pelajaran dari eksperimen 01–03 yang dipakai dalam desain
1. **Mulai sebagai identitas.** Setiap modul dibungkus `Bergerbang`: keluaran = x + g·(attn(x) − x), dengan g = 0 saat awal. Di langkah ke-0, model identik dengan A0 pralatih (diuji).
2. **SE pralatih tidak diganti.** Varian AL (SE → ECA) kehilangan bobot SE pralatih dan kalah (0,7263 vs 0,7301). Di dalam blok, attention ditambahkan *sesudah* SE:
   - CBAM-blok = SE pralatih sebagai bagian kanal + attention spasial CBAM yang baru. Urutan kanal → spasial sama dengan CBAM asli.
   - CA-blok = SE + CA.
3. **Grup optimizer benar.** Parameter attention di dalam `features` tetap masuk grup modul (lr 3·10⁻³; gerbang tanpa *weight decay*). Grup dasar (lr 3·10⁻⁴) hanya untuk bobot pralatih.

## Aturan (ditetapkan SEBELUM run penuh, 9 Nov 2026)
1. Strategi latih: S6-logit-adj. Keputusan utama: +bias_val (sama dengan eksperimen 03).
2. **Varian terpilih** = BACC validasi dalam rata-rata tertinggi (argmax) di antara 4 varian attention. Fold uji tidak dipakai untuk memilih.
3. Hipotesis "attention > A0" **didukung** bila varian terpilih unggul ≥ SESOI 2 poin BACC (+bias_val) dengan p_holm < 0,05. Koreksi Holm diterapkan atas 8 perbandingan (4 varian × 2 keputusan).
4. Bila didukung, kemenangan **wajib dikonfirmasi** dengan split baru (seed ulangan 1 dan 2) sebelum diklaim. Alasannya, fold seed 0 sudah dipakai di 4 eksperimen (risiko *garden of forking paths*).
5. Semua varian dilaporkan, termasuk yang kalah. A0 harus sama persis dengan A0 eksperimen 03.

## Data
Sama dengan eksperimen 01–03:
- cache `data/olahan/ham10000-256`
- StratifiedGroupKFold 5 fold per lesi, seed 0
- validasi dalam 10% lesi

## Metode
| Kode | Isi | Letak | Param. tambahan |
|---|---|---|---|
| A0 | MobileNetV3-Large polos | – | 0 |
| CBAM-akhir | CBAM penuh (kanal r=16 + spasial 7×7) | peta 960×7×7 sebelum GAP | 116.319 |
| CBAM-blok | attention spasial CBAM sesudah SE | 8 blok ber-SE (4–6, 11–15) | 792 |
| CA-akhir | coordinate attention (r=32) | peta 960×7×7 sebelum GAP | 88.411 |
| CA-blok | coordinate attention sesudah SE | 8 blok ber-SE | 295.103 |

Semua hiperparameter sama dengan eksperimen 03 (20 epoch, epoch terakhir, augmentasi sama).

## Metrik
- Sama dengan eksperimen 03.
- Tambahan: nilai gerbang attention di akhir latih (`parameter_terpelajari`) dan *loss* latih per epoch, untuk mengecek *overfitting*. Di eksperimen 03, *loss* latih A0 = 0,022 di epoch 20.

## Cara menjalankan
```bash
uv run python tugas/pcd-lesi-kulit/eksperimen/04-attention/run.py --config config-cepat.yaml   # uji jalan
uv run python tugas/pcd-lesi-kulit/eksperimen/04-attention/run.py                               # penuh, ±1 jam
```

## Hasil
| Run | Mode | Catatan |
|---|---|---|
| `hasil/20261109-123719` | cepat (1 fold, 2 epoch, 1.500 citra) | Uji jalan; angkanya **tidak** untuk laporan. A0 = uji jalan eksperimen 02/03, identik (0,2029 / 0,3446). Gerbang bergerak 0,002–0,04 setelah ±34 langkah. |
| `hasil/20261109-123813` | penuh, 1×5 fold, 20 epoch | Run acuan. 0,93 jam. Basis commit `9eca953` + perubahan belum di-commit (`perubahan_belum_commit.patch`). |

### Run `hasil/20261109-123813` (5 fold uji, rata-rata ± SD)
Cek konsistensi: A0 = 0,7388 / 0,7522, sama persis dengan eksperimen 02 dan 03.

| Varian | BACC val (argmax) | BACC uji argmax | BACC uji +bias_val | Δ vs A0 (+bias) | Fold positif | p_holm | AUROC makro | F1 makro (argmax) |
|---|---|---|---|---|---|---|---|---|
| A0 | 0,7407 | 0,7388 ± 0,0212 | **0,7522 ± 0,0230** | – | – | – | 0,9575 | 0,6945 |
| CBAM-akhir | 0,7340 | 0,7263 ± 0,0151 | 0,7421 ± 0,0226 | −0,0101 | 0/5 | 0,293 | 0,9615 | 0,6932 |
| CBAM-blok | 0,7328 | 0,7339 ± 0,0243 | 0,7507 ± 0,0255 | −0,0015 | 2/5 | 1,000 | 0,9596 | 0,6956 |
| **CA-akhir** (terpilih) | **0,7350** | 0,7351 ± 0,0232 | 0,7304 ± 0,0286 | −0,0218 | 1/5 | 1,000 | 0,9601 | 0,6973 |
| CA-blok | 0,7287 | 0,7390 ± 0,0297 | 0,7401 ± 0,0144 | −0,0121 | 1/5 | 1,000 | 0,9598 | 0,7004 |

Selisih argmax vs A0: CBAM-akhir −0,0125; CBAM-blok −0,0049; CA-akhir −0,0037; CA-blok +0,0002. Semua p_holm 1,000.

Diagnostik latih (rata-rata 5 fold):

| Varian | *Loss* latih ep 20 | BACC val tertinggi | BACC val ep 20 | Gerbang akhir |
|---|---|---|---|---|
| A0 | 0,0225 | 0,749 (ep 12) | 0,741 | – |
| CBAM-akhir | 0,0211 | 0,752 (ep 10) | 0,734 | −0,30 s.d. −0,41 |
| CBAM-blok | 0,0229 | 0,753 (ep 12) | 0,733 | rata-rata −0,06…0,05 (rentang −0,21…0,31) |
| CA-akhir | 0,0216 | 0,745 (ep 11) | 0,735 | −0,38 s.d. −0,45 |
| CA-blok | 0,0226 | 0,745 (ep 12) | 0,729 | rata-rata −0,08…0,05 (rentang −0,30…0,20) |

## Kesimpulan
**Hipotesis "attention > A0" tidak didukung.**
- Varian terpilih menurut aturan (CA-akhir) justru 2,18 poin di bawah A0 (+bias_val), hanya positif di 1 dari 5 fold.
- Tidak ada varian attention yang mengalahkan A0. Yang paling dekat adalah CBAM-blok (−0,15 poin, 2/5 fold), dengan hanya 792 parameter tambahan.
- Karena tidak ada yang menang, konfirmasi dengan split baru (aturan 4) tidak diperlukan.

**Gerbang di peta akhir konsisten negatif (≈ −0,4).**
- Dengan rumus x + g·(x·a − x), gerbang negatif berarti keluaran = x·(1,4 − 0,4·a). Daerah yang *diredam* attention justru diperkuat.
- Jadi model belajar membalik attention, bukan memusatkan diri. Ini sejalan dengan temuan H5d di eksperimen 03: model memanfaatkan informasi di seluruh citra, termasuk di luar lesi.
- Penafsiran ini dugaan dan belum diuji. Bisa juga efeknya sekadar penskalaan fitur.

**Pola umum eksperimen 01–04.**
- Enam belas varian modifikasi tidak ada yang mengalahkan A0: 7 di eksperimen 01 (A0+P, AL, A1, A3, A3-L, A3-C, A4), 5 konfigurasi M3 di eksperimen 03, dan 4 varian attention di eksperimen 04.
- Semua model mencapai *loss* latih ≈ 0,02. BACC validasi memuncak di epoch 10–12, lalu turun sampai epoch 20.
- AUROC naik sedikit (+0,002–0,004; tidak diuji), tetapi BACC tidak.
- Hambatan utamanya adalah generalisasi pada data latih yang terbatas, bukan kurangnya modul.

Keterbatasan:
- 1×5 fold, dengan fold yang sama dipakai di 4 eksperimen.
- Hiperparameter tidak di-*tuning* untuk attention (anggaran tuning sama dengan A0).
