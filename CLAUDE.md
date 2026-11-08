# Workspace Pengolahan Citra

Workspace untuk tugas kuliah dan eksperimen Pengolahan Citra Digital (Magister Informatika ITS), dikerjakan dengan Claude Code sebagai agen.

## Bahasa
- Komunikasi dan hasil tugas dalam **Bahasa Indonesia** baku (PUEBI), kecuali tugas meminta bahasa lain.
- Istilah teknis asing ditulis miring atau diberi padanan bila ada. Nama variabel/fungsi kode boleh bahasa Inggris atau Indonesia, asal konsisten per file.

## Struktur folder
- `tugas/<kode-matkul>-<nama-tugas>/` — satu folder per tugas:
  - `SOAL.md` — soal/instruksi asli dosen (jangan diubah)
  - `RENCANA.md` — rencana pengerjaan + checklist
  - `catatan-riset.md` — temuan riset beserta sumbernya
  - `eksperimen/<nn>-<nama>/` — satu folder per eksperimen (dibuat dari `template/eksperimen/` lewat `/eksperimen-baru`)
  - `hasil/` — jawaban akhir (laporan, slide, kode final)
- `citra/` — paket Python bersama: baca/simpan citra, metrik (MSE, PSNR, SSIM, akurasi), visualisasi, seed.
- `tests/` — tes untuk `citra/` (jalankan `uv run pytest`).
- `data/mentah/` — dataset asli (tidak di-commit). `data/olahan/` — hasil praproses (tidak di-commit).
- `referensi/` — PDF jurnal, slide, materi kuliah.
- `riset/` — validasi ide, rumusan masalah, dan keputusan arah riset (`riset/VALIDASI-IDE.md`); rencana kerja bertahap Plan 0–4 di `riset/PLAN.md`; proposal v1 `riset/PROPOSAL.md`; notulen sidang ide `riset/PLAN0-SIDANG.md`.
- `template/` — template makalah dan eksperimen.

## Referensi utama (wajib dibaca sebelum riset/eksperimen)
Ringkasan lengkap, angka, dan analisis kritis ada di `referensi/RINGKASAN-REFERENSI-UTAMA.md`. Teks penuh ada di `referensi/*.txt`.
- Shetty dkk. (2022), *Scientific Reports* 12:18134, doi:10.1038/s41598-022-22644-9 — fitur global (histogram warna, Hu moments, Haralick) + 7 klasifier ML vs CNN kustom pada subset seimbang HAM10000 (klaim 95,18% = akurasi latih; uji 86,43%).
- Lilhore dkk. (2024), *Scientific Reports* 14:4299, doi:10.1038/s41598-024-54212-8 — Hybrid U-Net + Improved MobileNet-V3 + Bayesian optimization pada HAM10000 (klaim akurasi 98,86%; ada inkonsistensi data/metrik yang sudah dicatat).
- Saat mengutip angka dari paper, ambil dari ringkasan/teks asli beserta nomor tabel/halaman; bedakan klaim penulis dari analisis kita.

## Lingkungan
- Python 3.12 dikelola `uv`. Pasang: `uv sync`. Jalankan apa pun dengan `uv run ...`.
- Library inti: numpy, opencv-python-headless, scikit-image, scikit-learn, matplotlib, pandas, pyyaml, jupyter.
- Deep learning (torch, torchvision) opsional: `uv sync --extra dl`.
- Jangan menambah dependensi tanpa alasan; bila perlu, `uv add <paket>` dan catat alasannya di README eksperimen.

## Aturan eksperimen (wajib)
- **Reproducible**: semua parameter di `config.yaml`, seed ditetapkan (`citra.seed_semua`), versi dataset/split dicatat.
- **Jangan mengarang angka.** Setiap angka di laporan/slide harus berasal dari `hasil/<run>/metrics.json` yang benar-benar dihasilkan skrip. Bila belum dijalankan, tulis `[BELUM DIJALANKAN]`.
- Selalu bandingkan dengan **baseline** dan laporkan metrik pada data uji yang terpisah (bedakan train/val/test).
- Setiap run menyimpan `config.yaml` salinan, `metrics.json`, dan gambar contoh keluaran di folder run-nya.
- Data mentah tidak pernah ditimpa; praproses menulis ke `data/olahan/`.

## Tim subagent (`.claude/agents/`)
- `peneliti` — cari & rangkum sumber, catat sitasi lengkap.
- `penulis` — susun makalah/laporan dari rencana, catatan riset, dan hasil eksperimen.
- `programmer` — implementasi algoritma citra & skrip eksperimen, lengkap dengan tes.
- `pemeriksa` — cek hasil terhadap soal: kelengkapan, kebenaran, sitasi, reproducibility.

## Alur kerja (slash command)
1. `/tugas-baru` — buat folder tugas, simpan soal, buat rencana.
2. `/eksperimen-baru` — buat folder eksperimen dari template di dalam tugas.
3. `/kerjakan` — jalankan rencana dengan subagent yang sesuai.
4. `/periksa` — review akhir sebelum dikumpulkan.
- Presentasi: pakai skill `presentasi-its` (butuh file template deck ITS di `referensi/` atau `template/`).

## Aturan akademik (wajib)
- **Jangan mengarang sumber.** Setiap kutipan/fakta harus punya sumber nyata (penulis, tahun, judul, jurnal, DOI/URL). Jika tidak yakin, tandai `[PERLU SUMBER]`.
- Gaya sitasi default **APA edisi 7**. Format angka Indonesia di tulisan (86,43%; 10.015).
- Hasil adalah **draf untuk dipelajari dan disunting pengguna**. Di akhir setiap tugas, beri ringkasan konsep kunci agar siap ditanya dosen.
- Patuhi aturan dosen/kampus tentang penggunaan AI.
