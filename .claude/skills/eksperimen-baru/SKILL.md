---
name: eksperimen-baru
description: Buat folder eksperimen pengolahan citra baru dari template (config.yaml, run.py, README) di dalam folder tugas.
argument-hint: "[folder-tugas] [nama-eksperimen]"
---

Argumen: $ARGUMENTS (folder tugas lalu nama eksperimen; jika folder tugas kosong, pakai folder di `tugas/` yang paling baru diubah; jika tidak ada tugas, tanyakan atau buat via `/tugas-baru`).

1. Tentukan nomor urut berikutnya `nn` (01, 02, ...) di `<folder-tugas>/eksperimen/` dan nama huruf kecil bertanda hubung.
2. Salin `template/eksperimen/` ke `<folder-tugas>/eksperimen/<nn>-<nama>/`.
3. Tanyakan singkat bila belum jelas dari `SOAL.md`/`RENCANA.md`: pertanyaan/hipotesis, dataset, baseline, metode kandidat, metrik.
4. Isi `README.md` (hipotesis, data, metode, metrik) dan sesuaikan `config.yaml`. Ubah `run.py` hanya bila eksperimennya bukan perbandingan filter — pertahankan pola: seed → `folder_run` → proses → `simpan_metrik`.
5. Jalankan sekali dengan konfigurasi kecil (`uv run python <folder>/run.py`) untuk memastikan berjalan; laporkan path hasil dan angka ringkasnya.
6. Tambahkan item checklist eksperimen ini di `RENCANA.md` tugas.
