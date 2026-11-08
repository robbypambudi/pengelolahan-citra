---
name: kerjakan
description: Kerjakan tugas Pengolahan Citra sesuai RENCANA.md dengan mendelegasikan langkah ke subagent peneliti, programmer, penulis, dan pemeriksa.
argument-hint: "[folder-tugas]"
---

Kerjakan tugas di folder: $ARGUMENTS (jika kosong, pakai folder di `tugas/` yang paling baru diubah; jika ragu, tanyakan).

1. Baca `SOAL.md` dan `RENCANA.md`. Jika belum ada rencana, jalankan dulu alur `/tugas-baru`.
2. Jalankan checklist berurutan; delegasikan ke subagent sesuai label:
   - riset → `peneliti` (hasil ke `catatan-riset.md`)
   - kode/eksperimen → `programmer` (eksperimen di `eksperimen/<nn>-<nama>/`, kode final di `hasil/`)
   - tulisan → `penulis` (hasil ke `hasil/`; angka hanya dari `metrics.json` eksperimen)
   - presentasi → skill `presentasi-its`
   Langkah yang saling independen boleh paralel. Selalu berikan path folder tugas lengkap di prompt subagent.
3. Setelah tiap langkah selesai, centang `- [x]` di `RENCANA.md`.
4. Setelah semua selesai, panggil `pemeriksa`. Perbaiki temuan serius, lalu periksa ulang sekali.
5. Akhiri dengan laporan ke pengguna:
   - File hasil yang siap dikumpulkan dan run eksperimen yang menjadi dasar angka
   - Sisa `[PERLU SUMBER]` / `[BELUM DIJALANKAN]` atau hal yang perlu dicek manual
   - **Ringkasan konsep kunci** (5–10 poin) agar pengguna paham isi tugasnya
