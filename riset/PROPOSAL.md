# Proposal Penelitian (v1 — hasil sidang Plan 0)

**Judul:** Seberapa Besar Kontribusi Nyata Pengolahan Citra pada Klasifikasi Lesi Kulit? Dekomposisi Inflasi Evaluasi dan Ablasi Praproses pada HAM10000

**Pengusul:** Robby Ulung Pambudi (NRP 6025252022) · Magister Informatika ITS · Mata kuliah Pengolahan Citra
**Status:** draf untuk disetujui (Gate 0) · 21 Oktober 2026 · notulen perdebatan: `riset/PLAN0-SIDANG.md`

---

## 1. Latar belakang
Klasifikasi lesi kulit dari citra dermoskopi pada HAM10000 (10.015 citra, 7 kelas) banyak melaporkan akurasi >95%. Dua paper rujukan utama:
- **Shetty dkk. (2022)**, *Sci. Rep.* 12:18134: mengklaim 95,18%, tetapi itu akurasi **latih**. Akurasi uji 86,43%, dan sekitar 85,0% bila dihitung ulang dari Tabel 7.
- **Lilhore dkk. (2024)**, *Sci. Rep.* 14:4299: mengklaim 98,86%.

Keduanya melakukan augmentasi/oversampling sebelum split, membagi data per citra, menguji pada distribusi buatan yang seimbang, dan memuat angka yang tidak konsisten. Contoh: confusion matrix Lilhore berjumlah 11.734 sampel, padahal porsi uji 15% hanya ±2.850 citra. Tabel 5 Shetty melanggar identitas recall tertimbang = akurasi. Sebagai pembanding, juara ISIC 2018 pada data uji resmi mencapai **BACC 0,885** (Codella dkk., 2019).

Masalah split per citra dan duplikat sudah dikenal (LesionBench; Cassidy dkk., 2022). Yang **belum** diketahui ada dua: (a) berapa kontribusi **masing-masing** pelanggaran protokol terhadap angka yang dilaporkan, dan (b) tahap pengolahan citra mana yang masih bermanfaat bila dievaluasi dengan benar.

## 2. Pertanyaan penelitian
- **RQ1, dekomposisi inflasi.** Pada implementasi yang sama, berapa poin BACC yang hilang pada setiap koreksi protokol berikut?
  - L0: protokol versi paper
  - L1: melaporkan akurasi uji, bukan latih
  - L2: augmentasi setelah split
  - L3: split per lesi
  - L4: uji pada distribusi asli
  - L5: data latih penuh (khusus Shetty)
- **RQ2, ablasi praproses.** Berapa efek utama dan interaksi dari penghapusan rambut (DullRazor), *color constancy* (Shades of Gray), dan CLAHE terhadap BACC dan sensitivitas melanoma, pada model berbasis fitur handcrafted dan pada CNN ringan?
- **RQ3, representasi.** Dengan protokol dan anggaran tuning yang sama, bagaimana fitur handcrafted (histogram HSV + Hu moments + Haralick) dibanding embedding MobileNetV3 dan fusi keduanya?
- **RQ4, segmentasi (opsional).** Bagaimana citra penuh dibanding crop dengan margin dan latar dihapus, dengan segmentasi Otsu+morfologi vs U-Net?

## 3. Hipotesis (dibekukan; SESOI = 2 poin BACC)
- **H1.** Total inflasi (L0 − L4) ≥ 5 poin BACC pada kedua pipeline. Untuk Shetty, komponen terbesar adalah L1 (akurasi latih) dan L2 (augmentasi sebelum split).
- **H2a.** Efek utama DullRazor pada seluruh data < SESOI (uji ekuivalensi), tetapi lebih besar pada strata citra berambut.
- **H2b (eksploratif).** Efek *color constancy* lebih besar pada partisi eksternal test (316 citra) daripada partisi internal (1.196). Ini uji konfirmasi temuan Barata dkk. (2015) pada CNN.
- **H2c.** Praproses tidak menurunkan sensitivitas melanoma lebih dari SESOI. Ini sekaligus menguji risiko CC menghapus ciri warna diagnostik.
- **H3.** Fitur handcrafted saja < CNN dengan selisih ≥ SESOI. Fusi tidak lebih buruk dari CNN (*non-inferior*). BACC handcrafted pada protokol bersih jauh di bawah 87% yang dilaporkan Shetty.

## 4. Desain singkat
- **Data:** HAM10000 + metadata `lesion_id` sebagai data latih/CV. ISIC 2018 Task 3 Test (1.512 citra) sebagai uji hold-out resmi, termasuk partisi eksternal 316 citra bila label sumbernya tersedia. ISIC 2018 Task 1 untuk mask U-Net, setelah citra yang beririsan dikeluarkan. Duplikat dideteksi dengan *perceptual hash* dan digabung ke satu grup.
- **Protokol:** StratifiedGroupKFold 3×5 berkelompok per lesi. Augmentasi dan normalisasi hanya dari data latih. Tuning hanya di fold validasi dalam dengan anggaran yang sama untuk semua pipeline. Data uji dipakai sekali.
- **Pipeline:**
  - P-S: re-implementasi Shetty (fitur global + 7 klasifier; CNN 96×96)
  - P-L: re-implementasi Lilhore (segmentasi → crop → MobileNetV3-Large → Bayesian optimization), dengan tabel kesetiaan
  - P-F: fusi
- **Metrik:**
  - Utama: BACC
  - Sekunder: sensitivitas melanoma @ spesifisitas 80%, F1 makro, AUROC makro, ECE
  - Segmentasi: Dice/IoU dan laju kegagalan
- **Statistik:**
  - Antar-fold: corrected resampled t-test (Nadeau & Bengio, 2003)
  - Faktorial RQ2: model linear efek campuran
  - Data uji: McNemar + bootstrap 95% CI
  - Klaim "tidak berbeda": TOST
  - Perbandingan ganda: koreksi Holm
  - Analisis sensitivitas tingkat lesi
- **Reprodusibilitas:** seed, `uv.lock`, prediksi per citra tersimpan, semua angka dihasilkan skrip, protokol dibekukan dengan git tag.

## 5. Kontribusi yang diharapkan
1. Angka **dekomposisi inflasi** per jenis pelanggaran protokol pada dua pipeline yang terbit, sebagai patokan untuk menilai klaim akurasi tinggi pada HAM10000.
2. **Ablasi per langkah** praproses pengolahan citra di bawah protokol bersih, termasuk analisis per strata kerambutan dan per kelas.
3. Perbandingan adil handcrafted vs CNN vs fusi.
4. Kode, file split, dan konfigurasi terbuka sebagai *baseline* yang dapat direproduksi.

## 6. Cakupan bertingkat
| Tingkat | Isi | Untuk |
|---|---|---|
| **MVP** | RQ1 pada P-S + RQ2 (RF-handcrafted & MobileNetV3) + uji hold-out 1.512 | Tugas kuliah (wajib) |
| **Inti** | + RQ1 pada P-L (Otsu & U-Net) + RQ3 | Tugas (bila waktu cukup) / dasar publikasi |
| **Publikasi** | + RQ4, uji lintas domain ISIC 2019/PH2, ulangan tambahan, rilis kode | Naskah jurnal |

## 7. Risiko utama & mitigasi
| Risiko | Mitigasi |
|---|---|
| Re-implementasi dianggap tidak setia | Inflasi diukur di dalam implementasi yang sama; tabel kesetiaan; L0 dibandingkan dengan angka paper |
| Label sumber test ISIC 2018 tidak tersedia | H2b dipindah ke tahap publikasi (ISIC 2019) |
| MDE > SESOI (desain kurang kuat) | Pilot mengukur simpangan baku dulu; ulangan ditambah sebelum protokol dibekukan |
| Komputasi RQ2 CNN berat (perkiraan 20–40 jam GPU) | Verifikasi di pilot; bila perlu turun ke 1×5 CV dan dinyatakan sebagai keterbatasan |
| Kritik terhadap paper terbit disalahpahami | Rujukan per tabel/gambar, hitung ulang ditampilkan, penulis asli dihubungi sebelum publikasi |

## 8. Pertanyaan untuk dosen
1. Berapa tenggat dan format luaran (laporan, presentasi, kode)?
2. Apakah studi **evaluasi ulang/replikasi + ablasi** diterima sebagai tugas Pengolahan Citra, atau harus ada "metode usulan baru"?
3. Bolehkah memakai model pralatih (MobileNetV3 ImageNet), atau ada batasan library/implementasi manual?
4. Apakah ada kebijakan departemen soal publikasi kritik terhadap paper lain dan soal penggunaan dataset CC BY-NC?

## Daftar pustaka (sementara; diverifikasi lengkap di Plan 2)
- Barata, C., Celebi, M. E., & Marques, J. S. (2015). Improving dermoscopy image classification using color constancy. *IEEE Journal of Biomedical and Health Informatics, 19*(3), 1146–1152.
- Cassidy, B., Kendrick, C., Brodzicki, A., Jaworek-Korjakowska, J., & Yap, M. H. (2022). Analysis of the ISIC image datasets: Usage, benchmarks and recommendations. *Medical Image Analysis, 75*, 102305. https://doi.org/10.1016/j.media.2021.102305
- Codella, N., dkk. (2019). Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the International Skin Imaging Collaboration (ISIC). arXiv:1902.03368.
- Lilhore, U. K., dkk. (2024). A precise model for skin cancer diagnosis using hybrid U-Net and improved MobileNet-V3 with hyperparameters optimization. *Scientific Reports, 14*, 4299. https://doi.org/10.1038/s41598-024-54212-8
- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*(3), 239–281. `[verifikasi DOI di Plan 2]`
- Shetty, B., dkk. (2022). Skin lesion classification of dermoscopic images using machine learning and convolutional neural network. *Scientific Reports, 12*, 18134. https://doi.org/10.1038/s41598-022-22644-9
- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset. *Scientific Data, 5*, 180161. https://doi.org/10.1038/sdata.2018.161
- Tschandl, P., dkk. (2019). Comparison of the accuracy of human readers versus machine-learning algorithms for pigmented skin lesion classification. *The Lancet Oncology, 20*(7), 938–947. https://doi.org/10.1016/S1470-2045(19)30333-X
