# Validasi Silang Ide Penelitian

Tanggal: 21 Oktober 2026 · Basis: dua referensi utama (lihat `referensi/RINGKASAN-REFERENSI-UTAMA.md`) + penelusuran literatur.
Tujuan: memastikan celah penelitian **nyata**, **belum ditutup** orang lain, dan **hasilnya dibutuhkan**, sebelum eksperimen dirancang.

---

## 1. Ide awal (titik tolak)
Kedua referensi (Shetty dkk., 2022; Lilhore dkk., 2024) melaporkan akurasi tinggi pada HAM10000, tetapi keduanya punya pola masalah yang sama:
1. augmentasi/oversampling dilakukan **sebelum** split
2. split **per citra**, bukan per lesi
3. evaluasi pada distribusi buatan yang seimbang
4. angka tidak konsisten antar tabel
5. kode tidak dibuka

Ide mentahnya adalah "memperbaiki metode dan evaluasi klasifikasi lesi kulit". Ide ini terlalu umum, jadi diuji di bawah ini.

## 2. Hasil validasi per klaim

| # | Klaim celah | Status di literatur | Putusan |
|---|---|---|---|
| C1 | Split per citra menggelembungkan performa di HAM10000 | **Sudah diketahui.** Hanya ±7.470 lesi unik dari 10.015 citra. Pada split acak, ±34–35% citra uji punya "saudara" (lesi sama) di data latih. Inflasi balanced accuracy +0,05 s/d +0,08 (LesionBench). Ada juga audit pratinjau yang menemukan akurasi turun dari 94,57% ke ±78% setelah koreksi per lesi. | ❌ **Bukan kontribusi baru.** Dipakai sebagai **protokol wajib**, bukan sebagai temuan. |
| C2 | Duplikat citra di dataset ISIC | **Sudah diketahui** (Cassidy dkk., 2022: duplikat di dalam dan antar dataset, termasuk lintas latih–uji). | ❌ Bukan kontribusi. Dipakai sebagai langkah pembersihan data. |
| C3 | "Mengalahkan akurasi 98,86%" | Angka itu kemungkinan besar hasil kebocoran. Acuan jujur adalah **balanced accuracy di data uji resmi ISIC 2018** (1.512 citra). Juaranya **BACC 0,885** (terkonfirmasi dari Codella dkk., 2019, Tabel 2). Test terdiri atas 1.196 citra internal + 316 citra eksternal. | ❌ **Target yang tidak valid.** Jangan mengejar akurasi; kejar estimasi yang jujur. |
| C4 | Belum ada yang memeriksa ulang (*replicate/audit*) Shetty 2022 atau Lilhore 2024 | Tidak ditemukan *matters arising*, replikasi, atau kritik terhadap keduanya. Lilhore hanya dikutip sebagai daftar pustaka. | ✅ **Celah spesifik, nyata, kontribusi sedang.** Hasilnya bisa diverifikasi karena data publik. |
| C5 | Kontribusi **tiap** tahap praproses citra (hapus rambut, *color constancy*) belum terisolasi | Studi yang ada **menggabungkan** langkah praproses (DullRazor + Gray World + CLAHE + median), sehingga efek tiap langkah tidak bisa dipisahkan. Bukti tentang menghapus artefak juga campur ("friend or foe?"). Tidak ditemukan ablasi per langkah dengan protokol bebas kebocoran **dan** uji eksternal. | ✅ **Celah terbuka.** Paling sesuai dengan mata kuliah Pengolahan Citra. |
| C6 | Apakah segmentasi/cropping lesi membantu klasifikasi | Satu studi terkontrol (ISIC 2017, biner, 2020): mask **tidak** signifikan membantu melanoma. *Dilated crop* membantu SK. Menghapus latar justru **merugikan**. Belum ada replikasi untuk 7 kelas HAM10000 dengan split per lesi. Lilhore mengklaim U-Net menaikkan akurasi 85 → 92%, tapi dengan protokol yang bocor. | ✅ **Celah terbuka**, dan langsung menguji klaim inti Lilhore. |
| C7 | Fitur *handcrafted* (warna, bentuk, tekstur) vs CNN vs fusi pada protokol yang sama | Tidak ditemukan perbandingan langsung ketiganya dengan balanced accuracy pada split yang sama. Studi fusi yang ada melaporkan akurasi biasa (98,4%) tanpa baseline CNN-only yang setara. | ✅ **Celah terbuka.** Langsung menguji klaim inti Shetty (RF 87% dari fitur global). |
| C8 | Generalisasi ke data luar | Pergeseran domain di dermoskopi sudah banyak dibahas (UDA, lokasi akuisisi). Untuk kita yang penting: **label uji ISIC 2018 Task 3 tersedia** (1.512 citra, label sama 7 kelas) dan **label uji ISIC 2019 tersedia** (8.238 citra, tetapi kelasnya berbeda sehingga perlu pemetaan). Label uji ISIC 2020 tidak tersedia. | ✅ **Bisa dilakukan.** Uji eksternal memperkuat validitas semua klaim di atas. |
| C9 | Arsitektur usulan Derm-MobileNetV3: MobileNetV3-Large + prior pengolahan citra yang bisa dilatih (M1 CC/rambut, M2 tekstur orde-2, M3 *pooling* lesi–kulit sekitar). Ditambahkan 8 November 2026 | Penelusuran `riset/LITERATUR-DERM-MOBILENETV3.md`: M3 **sebagian** (DermaKNet, PLCRD), M1 **sebagian** (SoG ≡ GeM; Hammouda dkk., 2026), M2 **sudah ada** (BILSK, SPCB-Net). | ✅ **Celah terbuka pada kombinasinya**, dengan M3 sebagai kontribusi utama dan M2 bukan klaim kebaruan. Menjadi RQ5 di `riset/PROPOSAL.md`. Sitasi maju di Google Scholar belum dicek. |

## 3. Rumusan yang lolos validasi

**Judul kerja:**
*Seberapa besar kontribusi nyata tahapan pengolahan citra pada klasifikasi lesi kulit dermoskopi? Evaluasi ulang bebas kebocoran terhadap pipeline Shetty dkk. (2022) dan Lilhore dkk. (2024).*

**Pertanyaan penelitian:**
- **RQ1 (audit):** Berapa performa sebenarnya pipeline kedua referensi bila diimplementasikan ulang dengan split per lesi, augmentasi hanya di data latih, dan diuji pada distribusi asli serta data uji resmi ISIC 2018? Seberapa besar selisihnya dengan angka yang mereka laporkan?
- **RQ2 (praproses):** Berapa kontribusi masing-masing hapus rambut (DullRazor), *color constancy* (Shades of Gray / Gray World), dan CLAHE, baik sendiri maupun gabungan, terhadap balanced accuracy dan recall melanoma?
- **RQ3 (representasi):** Pada protokol yang sama, bagaimana fitur *handcrafted* (histogram HSV + Hu moments + Haralick, ala Shetty) dibanding CNN ringan (MobileNet-V3, ala Lilhore) dan fusi keduanya?
- **RQ4 (segmentasi, opsional):** Apakah cropping berbasis segmentasi (dengan margin) membantu, dan apakah menghapus latar merugikan, pada 7 kelas HAM10000?

**Hipotesis (ditulis sebagai pertanyaan yang diuji, bukan klaim):**
- H1: performa kedua pipeline turun jauh (orde ≥5 poin balanced accuracy) di bawah protokol bebas kebocoran.
- H2: *color constancy* membantu lebih besar pada uji eksternal daripada uji internal, karena menekan perbedaan alat dan pencahayaan.
- H3: fusi *handcrafted* + CNN sedikit lebih baik daripada CNN saja. Fitur *handcrafted* saja jauh lebih rendah daripada angka 87% yang dilaporkan Shetty.
- H4: *crop* dengan margin ≥ tanpa crop > menghapus latar.

## 4. Uji "valid & dibutuhkan"

| Kriteria | Terpenuhi? | Alasan |
|---|---|---|
| Celah nyata (bukan asumsi) | Ya | Inkonsistensi kedua paper terdokumentasi dengan nomor tabel/gambar, dan sebagian sudah dihitung ulang. |
| Belum ditutup orang lain | Ya untuk C4–C7 | Masalah split dan duplikat sudah dikenal, jadi dijadikan protokol, bukan klaim kebaruan. |
| Bisa diverifikasi orang lain | Ya | Data publik (HAM10000, ISIC 2018/2019). Kode, konfigurasi, dan seed akan dibuka. |
| Dibutuhkan | Ya | Angka >95% yang bocor memberi ekspektasi palsu untuk penerapan klinis. Peneliti butuh tahu tahap praproses **mana** yang benar-benar layak dipakai. Model ringan relevan untuk fasilitas kesehatan dengan sumber daya terbatas. |
| Sesuai mata kuliah | Ya | Intinya tahapan pengolahan citra: filter, morfologi, inpainting, ruang warna, segmentasi, fitur tekstur/bentuk. |
| Layak dikerjakan | Ya | RTX 3080 Ti 12 GB, 24 core, RAM 31 GB, disk ±939 GB kosong. MobileNet-V3 dengan 5 seed × beberapa konfigurasi realistis dalam hitungan jam per konfigurasi. |

## 5. Ancaman validitas & mitigasi
- **Replikasi tidak persis** (kode asli tidak ada, detail kurang). Implementasikan sesuai teks, catat setiap asumsi di README, dan sebut sebagai "re-implementasi" lalu bandingkan **tren**, bukan angka absolut.
- **Label U-Net:** HAM10000 tidak punya *mask* resmi. Pilihannya mask ISIC 2018 Task 1 (perlu dicek irisannya dengan HAM10000; jangan sampai lesi uji ikut melatih segmentasi) atau segmentasi klasik (Otsu + morfologi) yang lebih cocok untuk mata kuliah. **Perlu keputusan.**
- **Uji eksternal ISIC 2018 Test** mungkin berisi lesi dari sumber yang sama dengan HAM10000. Periksa duplikat dengan *perceptual hash* sebelum dipakai.
- **Pemetaan label ISIC 2019** (8 kelas, ada SCC, akiec tidak sama persis). Pakai hanya kelas yang bisa dipetakan, atau laporkan secara terpisah.
- **Varian antar-run.** Gunakan ≥3–5 seed dan laporkan rerata ± simpangan baku, serta uji signifikansi berpasangan (mis. McNemar atau bootstrap CI) untuk klaim "lebih baik".
- **Metrik:** utamakan balanced accuracy, F1 makro, recall per kelas (terutama melanoma), dan AUROC makro. Akurasi biasa dilaporkan sebagai pelengkap saja.
- **Angka sekunder yang belum dikonfirmasi** (±7.470 lesi unik, juara ISIC 2018 88,5%, audit pratinjau 94,57 → 78%). Hitung ulang jumlah lesi dari metadata setelah data diunduh, dan cek leaderboard di makalah resmi ISIC 2018 (arXiv 1902.03368).

## 6. Keputusan pengguna (21 Okt 2026)
1. **Cakupan:** RQ1 + RQ2 + RQ3; RQ4 opsional.
2. **Segmentasi:** klasik (Otsu + morfologi) **dan** U-Net (mask ISIC 2018 Task 1) dibandingkan.
3. **Luaran:** tugas kuliah dulu, lalu diperluas ke publikasi.

Rencana kerja lengkap: `riset/PLAN.md`.

## Sumber
- Cassidy, B., Kendrick, C., Brodzicki, A., Jaworek-Korjakowska, J., & Yap, M. H. (2022). Analysis of the ISIC image datasets: Usage, benchmarks and recommendations. *Medical Image Analysis, 75*, 102305. https://doi.org/10.1016/j.media.2021.102305 (repositori: https://e-space.mmu.ac.uk/631025)
- Codella, N. dkk. (2019). Skin lesion analysis toward melanoma detection 2018: A challenge hosted by ISIC. arXiv:1902.03368. https://arxiv.org/abs/1902.03368
- ISIC Challenge Data (unduhan resmi 2018/2019/2020). https://challenge.isic-archive.com/data/
- LesionBench (repositori GitHub, inflasi split per citra vs per lesi). https://github.com/VivekReddy2001/lesionbench
- Audit kebocoran HAM10000 pada model hibrida kuantum (pratinjau ResearchGate; dipakai hati-hati). https://www.researchgate.net/publication/408147757
- Tutorial Galaxy "Validating Skin Lesion Classification on HAM10000". https://training.galaxyproject.org/training-material/topics/statistics/tutorials/image_learner/tutorial.html
- Mahbod, A. dkk. (2020). The effects of skin lesion segmentation on the performance of dermatoscopic image classification. arXiv:2008.12602. https://arxiv.org/abs/2008.12602
- Dermoscopic dark corner artifacts removal: Friend or foe? arXiv:2306.13446. https://arxiv.org/abs/2306.13446
- Hair and artifact removal in dermoscopic images using deep learning (Scientific Reports, 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13149812/
- Mitigating the influence of domain shift in skin lesion classification (benchmark UDA). arXiv:2310.03432. https://arxiv.org/abs/2310.03432
- Fusi fitur handcrafted + MobileNet/ResNet101 (MDPI Processes 11(3):910). https://www.mdpi.com/2227-9717/11/3/910
- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset. *Scientific Data, 5*, 180161. https://doi.org/10.1038/sdata.2018.161

> Catatan: penulis/judul beberapa sumber di atas berasal dari hasil penelusuran dan belum dibaca penuh. Verifikasi metadata sitasinya sebelum dipakai di laporan akhir.
