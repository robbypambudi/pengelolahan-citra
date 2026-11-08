# Rencana Penelitian — Plan 0 s/d Plan 4

**Judul kerja:** Seberapa besar kontribusi nyata tahapan pengolahan citra pada klasifikasi lesi kulit dermoskopi? Evaluasi ulang bebas kebocoran terhadap pipeline Shetty dkk. (2022) dan Lilhore dkk. (2024).

**Keputusan yang sudah diambil (21 Okt 2026):**
- Cakupan: **RQ1 + RQ2 + RQ3**. RQ4 (segmentasi untuk klasifikasi) opsional.
- Segmentasi: **klasik (Otsu + morfologi) dan U-Net dibandingkan**. Keduanya tetap dibutuhkan di RQ1 untuk re-implementasi pipeline Lilhore, walaupun RQ4 opsional.
- Luaran: **tugas kuliah dulu, lalu diperluas ke publikasi**. Desain dibuat sejak awal agar bisa diperluas (seed, uji eksternal, kode rapi).

**Aturan main antar-plan:** setiap plan punya **gerbang (gate)**. Plan berikutnya baru dimulai bila gate plan sebelumnya lolos. Pengecualian: Plan 1 dan Plan 2 boleh berjalan paralel.

| Plan | Fokus | Luaran utama | Status |
|---|---|---|---|
| 0 | Brainstorming & validasi ide | `VALIDASI-IDE.md`, `PLAN0-SIDANG.md`, `PROPOSAL.md` | 🟢 selesai, menunggu persetujuan (Gate 0) |
| 1 | Dataset | `data/README.md`, `data/split/*.csv`, laporan EDA & duplikat | ⬜ |
| 2 | Studi literatur | `riset/LITERATUR.md` (tabel ekstraksi + matriks celah) | ⬜ |
| 3 | Metodologi | `riset/METODOLOGI.md` (protokol dibekukan) | ⬜ |
| 4 | Eksperimen | `tugas/.../eksperimen/E1–E4`, laporan, slide | ⬜ |

---

## Plan 0 — Brainstorming & Validasi Ide

**Tujuan:** memastikan ide punya celah yang nyata, belum ditutup orang lain, dibutuhkan, dan layak dikerjakan.

Sudah dikerjakan:
- [x] Membaca penuh dua referensi utama dan mencatat inkonsistensinya (`referensi/RINGKASAN-REFERENSI-UTAMA.md`)
- [x] Validasi silang ide ke literatur: kebocoran per citra dan duplikat ISIC **sudah dikenal**, jadi dijadikan protokol, bukan kontribusi (`riset/VALIDASI-IDE.md`)
- [x] Menetapkan celah: audit kedua pipeline, ablasi praproses per langkah, *handcrafted* vs CNN vs fusi, efek segmentasi
- [x] Cek kelayakan komputasi (RTX 3080 Ti 12 GB, 24 core, RAM 31 GB)
- [x] Keputusan cakupan, segmentasi, dan luaran

- [x] Sidang ide (profesor vs pengusul, 15 babak) → `riset/PLAN0-SIDANG.md`
- [x] Proposal revisi v1 → `riset/PROPOSAL.md`. Perubahan utama: RQ1 menjadi tangga dekomposisi L0–L5, SESOI 2 pp + TOST, 3×5 CV + uji Nadeau-Bengio, stratifikasi kerambutan, metrik klinis sensitivitas melanoma @ spesifisitas 80%, cakupan MVP/inti/publikasi
- [x] Konfirmasi dari sumber primer: test ISIC 2018 = 1.196 internal + 316 eksternal; BACC teratas 0,885 (Codella dkk., 2019)

Sisa (dibawa ke plan berikutnya):
- [ ] ❓ Tenggat, format, dan penerimaan topik → tanya dosen (`PROPOSAL.md` §8)
- [ ] ❓ Label sumber & distribusi kelas test ISIC 2018; proporsi citra berambut → Plan 1 (jumlah lesi unik ✅ 7.470)
- [ ] ❓ Simpangan baku antar-fold, MDE, dan waktu per run → pilot Plan 4.1, sebelum Plan 3 dibekukan

### 0.R Replikasi Shetty dkk. (2022) apa adanya (protokol L0)
**Tujuan:** uji kelayakan. Apakah angka Shetty bisa direproduksi dengan protokol persis seperti paper? Hasilnya menjadi titik awal (L0) tangga dekomposisi RQ1 dan memvalidasi infrastruktur kode.
Lokasi: `tugas/pcd-lesi-kulit/eksperimen/00-replikasi-shetty-L0/`

- [x] Forensik Tabel 4: ke-14 akurasi tepat kelipatan 1/560 (tanpa augmentasi) dan 1/1120 (dengan flip). Artinya Tabel 4 = **akurasi 10-fold CV pada porsi latih 80%**, bukan akurasi uji.
- [x] Unduh HAM10000 (ISIC 2018 Task 3 Training + GroundTruth + LesionGroupings). Terverifikasi: 10.015 citra, **7.470 lesi unik** (1.956 lesi >1 citra, maksimal 6), distribusi kelas sama dengan Tschandl dkk. (2018)
- [x] Pipeline ML sesuai paper (RF + flip: **87,00 ± 0,67%** vs paper 87,32%; 78,6% uji punya kembaran di data latih):
  - subset 100 citra/kelas, resize 220×220
  - fitur histogram HSV 8×8×8 + Haralick (13) + Hu (7) = 532 dimensi
  - MinMaxScaler; flip horizontal → 200/kelas; split 80:20
  - 10-fold CV pada porsi latih; 7 klasifier dengan hiperparameter Tabel 1
- [x] Pipeline CNN sesuai paper (latih **95,42 ± 1,10%** vs 95,18%; uji 90,60 vs 86,43, selisih terdokumentasi): 96×96, arsitektur Tabel 2, Adam, 150 epoch, batch 32, LR 1e-3 → 1e-5, augmentasi acak, split 80:20 dari 1.400 citra (uji = 280)
- [x] Ulangi dengan beberapa seed (ML 10 seed subset, CNN 3 seed)
- [x] Bandingkan dengan Tabel 4–7 paper; 10 asumsi re-implementasi dicatat (README eksperimen 00)
- **Gate 0.R: LOLOS** (21 Okt 2026)

**Gate 0.R:** angka replikasi berada dalam rentang wajar dari angka paper, atau selisihnya dapat dijelaskan dan didokumentasikan.

**Gate 0:** proposal 1 halaman disetujui pengguna (dan dosen bila perlu).

---

## Plan 1 — Dataset

**Tujuan:** data yang bersih, terdokumentasi, dan split yang **dijamin bebas kebocoran** sebelum ada model yang dilatih.

### 1a. Pengadaan data (`data/mentah/`, tidak di-commit)
| Data | Isi | Dipakai untuk |
|---|---|---|
| HAM10000 / ISIC 2018 Task 3 Training | 10.015 citra + label 7 kelas | latih & validasi silang internal |
| Metadata HAM10000 (Harvard Dataverse) | `lesion_id`, `dx_type`, umur, jenis kelamin, lokasi, sumber | split per lesi, analisis subkelompok |
| ISIC 2018 Task 3 Test + Ground Truth | 1.512 citra | **uji eksternal**, dipakai sekali di akhir |
| ISIC 2018 Task 1 (citra + mask) | ±2.594 citra + mask lesi | melatih U-Net dan mengukur Dice segmentasi |
| ISIC 2019 Test + GT *(tahap publikasi)* | 8.238 citra, 8 kelas | uji lintas domain (perlu pemetaan label) |

- [ ] Unduh, verifikasi jumlah file dan checksum, dan catat URL, versi, tanggal, serta lisensi (CC BY-NC 4.0)

### 1b. Analisis data eksploratif (EDA)
- [ ] Distribusi kelas (cocokkan dengan 6.705 nv / 1.113 mel / … menurut Tschandl dkk., 2018)
- [x] **Hitung jumlah lesi unik dan citra per lesi**: 7.470 lesi, 1.956 lesi >1 citra, maksimal 6 (dikerjakan saat 0.R)
- [ ] Distribusi `dx_type`, sumber/klinik, lokasi tubuh, umur, dan jenis kelamin
- [ ] Profil artefak: perkiraan proporsi citra berambut, bersudut gelap, dan berpenggaris (sampel manual ±200 citra). Ini menjadi dasar RQ2.

### 1c. Pemeriksaan duplikat & irisan
- [ ] *Perceptual hash* (pHash/dHash) di dalam HAM10000 untuk menemukan duplikat dengan `lesion_id` berbeda
- [ ] HAM10000 vs ISIC 2018 Test, untuk memastikan uji eksternal benar-benar terpisah
- [ ] HAM10000 vs ISIC 2018 Task 1, untuk memastikan lesi uji tidak ikut melatih U-Net

### 1d. Protokol split (dibekukan sebagai file)
- [ ] Uji eksternal = ISIC 2018 Test, **tidak disentuh** sampai Plan 4 tahap akhir
- [ ] Internal: **StratifiedGroupKFold 3×5 (berulang, 3 seed split)** dengan grup = `lesion_id` (setelah duplikat digabung ke grup yang sama), plus fold validasi dalam untuk tuning. Lihat Sidang Babak 8: 5 fold saja tidak bisa mencapai p<0,05.
- [ ] Simpan `data/split/folds.csv` (image_id, lesion_id, dx, fold) dan commit
- [ ] Tes otomatis `tests/test_split.py`: tidak ada `lesion_id` atau hash duplikat yang muncul di dua fold, dan proporsi kelas per fold mirip

**Luaran:** `data/README.md` (*datasheet*), notebook EDA, `data/split/folds.csv`, laporan duplikat.
**Gate 1:** tes split lulus, laporan duplikat ada, dan angka EDA cocok dengan sumber resmi.

---

## Plan 2 — Studi Literatur

**Tujuan:** memposisikan setiap RQ terhadap penelitian terdahulu dan memastikan klaim kebaruan aman. Boleh paralel dengan Plan 1.

### 2a. Protokol penelusuran
- Basis data: Scopus/IEEE Xplore/ScienceDirect (lewat akses ITS), PubMed, arXiv, Google Scholar
- Rentang 2015–2026, ditambah karya klasik/asal metode tanpa batas tahun
- Kata kunci per topik (lihat 2b), dicatat beserta tanggal pencarian dan jumlah hasil

### 2b. Topik & sumber wajib
| Topik | Untuk RQ | Sumber kunci (minimal) |
|---|---|---|
| Protokol evaluasi, kebocoran, duplikat | RQ1 | Cassidy dkk. 2022; Codella dkk. 2019 (ISIC 2018); Tschandl dkk. 2018; studi split per lesi |
| Penghapusan rambut | RQ2 | DullRazor (Lee dkk., 1997); studi hair removal + DL |
| *Color constancy* | RQ2 | Shades of Gray (Finlayson & Trezzi, 2004); Barata dkk. 2015 (CC untuk dermoskopi) |
| Peningkatan kontras | RQ2 | CLAHE (Zuiderveld, 1994) |
| Fitur *handcrafted* | RQ3 | Hu (1962); Haralick dkk. (1973); aturan ABCD |
| CNN ringan & atensi | RQ1, RQ3 | MobileNetV3 (Howard dkk., 2019); ECA-Net (Wang dkk., 2020) |
| Segmentasi | RQ1, RQ4 | U-Net (Ronneberger dkk., 2015); Otsu (1979); Mahbod dkk. 2020 (efek segmentasi pada klasifikasi) |
| Fusi fitur | RQ3 | studi fusi handcrafted + deep pada dermoskopi |
| Statistik perbandingan model | semua | Demšar (2006); McNemar; bootstrap CI |

### 2c. Ekstraksi & sintesis
- [ ] Tabel ekstraksi per paper: dataset · level split (citra/lesi) · augmentasi sebelum/sesudah split · metrik (akurasi vs balanced) · uji eksternal? · kode terbuka? · hasil
- [ ] **Matriks celah**: baris = RQ, kolom = paper. Tunjukkan bahwa sel "split per lesi + uji eksternal + ablasi per langkah" masih kosong.
- [ ] Draf *related work* (±1–2 halaman) untuk laporan
- [ ] Verifikasi setiap sitasi (DOI benar-benar ada, metadata cocok). Tandai yang belum diverifikasi dengan `[PERLU SUMBER]`.

**Luaran:** `riset/LITERATUR.md`, `catatan-riset.md`, draf *related work*.
**Gate 2:** setiap RQ punya ≥3 karya terkait dengan posisi celah yang jelas, dan semua sitasi terverifikasi.

---

## Plan 3 — Metodologi (protokol dibekukan sebelum eksperimen utama)

**Tujuan:** menulis protokol lengkap **sebelum** melihat hasil, agar tidak ada penyesuaian setelah melihat angka (*p-hacking*). Mirip pra-registrasi.

### 3a. Prinsip protokol
- Split dari Plan 1. Augmentasi, oversampling, dan statistik normalisasi **hanya dari data latih fold tersebut**.
- Tuning hiperparameter hanya di fold validasi dalam. Uji eksternal dijalankan **sekali** dengan konfigurasi final.
- Setiap konfigurasi dijalankan pada 5 fold (+ ≥3 seed untuk model stokastik).

### 3b. Pipeline yang dibangun
| Kode | Pipeline | Isi |
|---|---|---|
| **P-S** | Re-implementasi Shetty | resize 220 → (flip hanya di latih) → histogram HSV + Hu moments + Haralick → 7 klasifier (hiperparameter Tabel 1); CNN kustom 96×96 (Tabel 2–3) |
| **P-L** | Re-implementasi Lilhore (versi realistis) | segmentasi (Otsu+morfologi **atau** U-Net) → crop lesi → MobileNetV3-Large pralatih ImageNet, 224×224, opsi SE→ECA → Bayesian optimization (Optuna, anggaran tetap) di fold validasi dalam |
| **P-F** | Fusi | embedding MobileNetV3 + fitur handcrafted → klasifier linear/MLP |

Setiap detail yang tidak dijelaskan paper asli dicatat sebagai **asumsi re-implementasi**.

### 3c. Desain eksperimen per RQ
| Eks. | RQ | Faktor yang divariasikan | Tetap |
|---|---|---|---|
| **E1** | RQ1 dekomposisi | tangga L0→L5 (satu perubahan per langkah: akurasi uji, augmentasi setelah split, split per lesi, distribusi asli, data penuh) + satu urutan alternatif × pipeline P-S, P-L | implementasi & hiperparameter yang sama |
| **E2** | RQ2 praproses | faktorial 2³: hapus rambut {0,1} × *color constancy* {0,1} × CLAHE {0,1} = 8 kombinasi × {RF-handcrafted, MobileNetV3} | split, epoch, seed |
| **E3** | RQ3 representasi | handcrafted vs CNN vs fusi | praproses terbaik dari E2 (atau tanpa praproses, ditentukan di protokol) |
| **E4** *(opsional)* | RQ4 segmentasi | input: citra penuh / crop margin 10–20% / latar dihapus × segmentasi Otsu vs U-Net | MobileNetV3, split |
| **E5** | RQ5 Derm-MobileNetV3 | A0, A0+P, AL, A1, A3, A3-L, A3-C, A4 (lihat `riset/PROPOSAL.md` §4) | hiperparameter, augmentasi, split, seed |

Untuk RQ4, kualitas segmentasi juga diukur (Dice/IoU di ISIC 2018 Task 1 bagian uji yang bebas irisan).

### 3d. Metrik & statistik
- **Utama:** balanced accuracy (= mean recall, metrik resmi ISIC 2018)
- **Sekunder:** F1 makro, **recall melanoma**, AUROC makro, ECE (kalibrasi). Akurasi biasa hanya pelengkap.
- Laporan: rerata ± simpangan baku antar-fold, confusion matrix dengan jumlah sampel yang cocok dengan ukuran split
- Uji signifikansi: corrected resampled t-test Nadeau-Bengio pada 3×5 CV (internal); model linear efek campuran untuk faktorial E2; McNemar dan bootstrap 95% CI (uji hold-out); TOST dengan SESOI 2 pp untuk klaim "tidak berbeda"; koreksi Holm untuk perbandingan ganda. MDE dihitung dari pilot.
- **Konsistensi:** semua metrik dihitung dari prediksi tersimpan (bukan disalin manual), sehingga kesalahan seperti Tabel 5 Shetty tidak mungkin terjadi

### 3e. Ancaman validitas
Re-implementasi tidak persis, irisan antar-dataset, pemetaan label ISIC 2019, varian antar-seed, dan bias subkelompok (lokasi tubuh/sumber klinik). Rincian dan mitigasinya ada di `riset/VALIDASI-IDE.md` §5.

**Luaran:** `riset/METODOLOGI.md`, diagram pipeline, daftar asumsi re-implementasi.
**Gate 3:** review oleh subagent `pemeriksa` dan persetujuan pengguna, lalu **protokol dibekukan** (git tag `protokol-v1`). Perubahan setelah ini dicatat sebagai deviasi beserta alasannya.

---

## Plan 4 — Eksperimen

**Tujuan:** menjalankan protokol Plan 3 secara reproducible dan menjawab RQ dengan angka dari `metrics.json`.

### 4.0 Infrastruktur
- [ ] Modul `citra/`: loader data + split, praproses (DullRazor, Shades of Gray/Gray World, CLAHE), fitur (histogram HSV, Hu, Haralick), segmentasi Otsu+morfologi
- [ ] Modul model: RF dan klasifier lain (sklearn), MobileNetV3 (torchvision), U-Net, fusi
- [ ] Pencatatan: config + seed + git commit + prediksi per citra (`preds.csv`) + `metrics.json`
- [ ] Tes unit: kebocoran split, praproses deterministik, dan metrik cocok dengan sklearn
- [ ] Pasang `uv sync --extra dl` dan cek GPU dari PyTorch

### 4.1 Uji coba (sanity check)
- [ ] Jalankan setiap pipeline pada 1 fold dengan 1–2 epoch dan data kecil untuk memastikan alur dan format hasil benar

### 4.2–4.5 Eksperimen utama (berurutan)
- [ ] **E1 Audit** — P-S dan P-L dengan protokol *as-reported* vs *terkoreksi*, lalu ukur besar inflasinya
- [ ] **E2 Ablasi praproses** — 8 kombinasi × 2 model × 5 fold
- [ ] **E3 Representasi** — handcrafted vs CNN vs fusi
- [ ] **E4 Segmentasi** *(opsional)* — latih U-Net, bandingkan dengan Otsu, lalu tiga mode input
- [ ] **E5 Derm-MobileNetV3 (RQ5)** — `tugas/pcd-lesi-kulit/eksperimen/01-derm-mobilenetv3/`
  - [x] Modul `citra/derm.py` (M1, M2, M3, varian AL) + tes identitas saat inisialisasi
  - [x] `citra/segmentasi.py` (Otsu) + `citra/split.py` (StratifiedGroupKFold per lesi) beserta tes
  - [x] Cache 256 px + mask Otsu (`siapkan.py`; 439/10.015 jatuh ke elips cadangan)
  - [x] Uji jalan `config-cepat.yaml` (8 varian, 1 fold, 2 epoch) → `hasil/20261108-215858`
  - [ ] Run penuh 8 varian × 5 fold (1×5), lalu 3×5 bila MDE menuntut
  - [ ] Mask U-Net (ISIC 2018 Task 1, irisan dikeluarkan) untuk sub-varian A3
  - [ ] Ulang dengan split v1 (dedup pHash) dan uji hold-out 1.512 citra

### 4.6 Uji eksternal (sekali)
- [ ] Konfigurasi final setiap RQ diuji pada ISIC 2018 Test (1.512 citra) dengan bootstrap CI
- [ ] *(Publikasi)* uji lintas domain ISIC 2019 dengan pemetaan label

### 4.7 Analisis
- [ ] Tabel dan grafik per RQ, uji signifikansi, dan jawaban H1–H4 (diterima/ditolak beserta angkanya)
- [ ] Analisis kesalahan: kelas yang sering tertukar (mel↔nv, bkl↔mel); Grad-CAM opsional
- [ ] Analisis subkelompok: sumber klinik dan lokasi tubuh

### 4.8 Penulisan & luaran
- [ ] **Tugas:** laporan (`template/makalah.md`) + presentasi gaya ITS (skill `presentasi-its`) dengan alur review → celah → metodologi → hasil → kesimpulan
- [ ] **Publikasi (lanjutan):** tambah seed dan ISIC 2019, buka kode dan split (GitHub), lalu tulis naskah. Calon target jurnal ditentukan nanti.

**Perkiraan komputasi (kasar, diverifikasi di 4.1):** satu run MobileNetV3 224 px pada ±8 rb citra sekitar 10–20 menit di RTX 3080 Ti. E2 (8 × 5 fold × 1 seed untuk CNN) ≈ 7–13 jam GPU. Pipeline handcrafted terutama dibatasi oleh CPU (24 core), dan fiturnya cukup diekstraksi sekali lalu disimpan ke cache.

**Gate 4:** semua angka di laporan dapat dilacak ke `metrics.json`, tidak ada `[BELUM DIJALANKAN]` yang tersisa, dan subagent `pemeriksa` lolos.

---

## Urutan & ketergantungan

```
Plan 0 ──► Plan 1 (dataset) ──┐
      └──► Plan 2 (literatur) ─┴──► Plan 3 (protokol beku) ──► Plan 4 (eksperimen) ──► Laporan/Slide ──► Publikasi
```

Jadwal per plan disusun setelah tenggat tugas dari dosen diketahui.
