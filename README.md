# Pengolahan Citra — Workspace Tugas & Eksperimen

## Persiapan

```bash
cd ~/pengelolaan_citra
uv sync                 # lingkungan inti (OpenCV, scikit-image, scikit-learn, ...)
uv sync --extra dl      # opsional: PyTorch + torchvision
uv run pytest           # cek paket citra/ berjalan
claude
```

## Perintah di Claude Code

| Perintah | Fungsi |
|---|---|
| `/tugas-baru PCD tugas-1` + tempel soal | Buat folder tugas, simpan soal, susun rencana |
| `/eksperimen-baru tugas/pcd-tugas-1 denoising-median` | Buat folder eksperimen dari template |
| `/kerjakan tugas/pcd-tugas-1` | Jalankan rencana memakai tim subagent |
| `/periksa tugas/pcd-tugas-1` | Review akhir sebelum dikumpulkan |

## Menjalankan eksperimen

```bash
uv run python tugas/pcd-tugas-1/eksperimen/01-denoising-median/run.py
# hasil: tugas/pcd-tugas-1/eksperimen/01-denoising-median/hasil/<waktu>/
#   config.yaml  metrics.json  contoh/*.png
```

## Struktur

```
citra/       paket bersama (io, metrik, visualisasi, seed)
tests/       tes paket citra
data/        mentah/ & olahan/ (tidak di-commit)
tugas/       satu folder per tugas (SOAL, RENCANA, catatan-riset, eksperimen/, hasil/)
referensi/   PDF/slide/materi dari dosen
template/    makalah.md, eksperimen/
```

> Hasil adalah draf untuk dipelajari dan disunting. Patuhi aturan kampus/dosen soal penggunaan AI.
