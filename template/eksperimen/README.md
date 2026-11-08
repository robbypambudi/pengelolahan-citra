# Eksperimen: <judul>

## Pertanyaan / hipotesis
<Apa yang diuji? Tulis sebagai pertanyaan yang bisa dijawab dengan angka, mis. "Apakah filter median menghasilkan PSNR lebih tinggi daripada mean untuk derau salt-and-pepper 5%?">

## Data
- Sumber & versi: <nama dataset, URL, lisensi>
- Lokasi: `data/mentah/<...>` · Split: <train/val/test, cara membagi, seed>

## Metode
- Baseline: <...>
- Kandidat: <...>
- Parameter: lihat `config.yaml`

## Metrik
<PSNR, SSIM, akurasi, F1 macro, ... — dan alasannya>

## Cara menjalankan
```bash
uv run python <path-folder-ini>/run.py
```

## Hasil
| Run | Metode | Metrik utama | Catatan |
|---|---|---|---|
| `hasil/<waktu>` | | `[BELUM DIJALANKAN]` | |

## Kesimpulan
<Jawab pertanyaan di atas berdasarkan angka di `hasil/`. Sebutkan keterbatasan.>
