---
name: tugas-baru
description: Mulai tugas kuliah Pengolahan Citra baru — buat folder tugas, simpan soal, dan susun rencana pengerjaan (termasuk eksperimen).
argument-hint: "[kode-matkul] [nama-tugas] (lalu tempel soalnya)"
---

Pengguna ingin memulai tugas baru. Argumen: $ARGUMENTS

1. Tentukan nama folder `tugas/<kode-matkul>-<nama-tugas>/` (huruf kecil, tanda hubung). Jika info kurang (matkul, soal, tenggat), tanyakan singkat.
2. Simpan soal persis seperti diberikan ke `SOAL.md`, ditambah metadata di atas: mata kuliah, dosen (jika ada), tenggat, jenis tugas (makalah/laporan/kode/eksperimen/presentasi/review artikel).
3. Jika soal berupa file di `referensi/` (PDF/gambar), baca dan salin isinya ke `SOAL.md`.
4. Buat `RENCANA.md` berisi:
   - Pemahaman soal (ringkas, dengan kata sendiri)
   - Kriteria penilaian / ketentuan (termasuk batasan library, mis. wajib implementasi manual)
   - Rancangan eksperimen bila ada: pertanyaan, dataset, baseline, metode, metrik
   - Langkah pengerjaan berupa checklist `- [ ]`, masing-masing ditandai subagent (peneliti / programmer / penulis / pemeriksa)
   - Pertanyaan terbuka untuk pengguna
5. Buat folder `hasil/` dan `eksperimen/` kosong.
6. Tampilkan ringkasan rencana dan minta persetujuan sebelum `/kerjakan`.
