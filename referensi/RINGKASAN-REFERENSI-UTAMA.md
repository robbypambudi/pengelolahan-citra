# Referensi Utama Proyek

Paper inti yang menjadi acuan proyek ini. Teks lengkap hasil ekstraksi ada di file `.txt` bernama sama, untuk dicari ulang (grep) bila perlu kutipan persis.

| Paper | File |
|---|---|
| Shetty dkk. (2022) — klasifikasi lesi kulit dengan ML (fitur global) dan CNN kustom (dibaca penuh, 11 hlm.) | `s41598-022-22644-9.pdf` |
| Lilhore dkk. (2024) — Hybrid U-Net + Improved MobileNet-V3 untuk diagnosis kanker kulit (dibaca penuh, 24 hlm.) | `s41598-024-54212-8.pdf` |

Keduanya memakai HAM10000 dan terbit di *Scientific Reports*. Lilhore dkk. bahkan mengutip Shetty dkk. sebagai referensi no. 19.

Konvensi: **[Fakta]** = isi paper (dengan halaman); **[Analisis]** = interpretasi/temuan kita sendiri, bukan klaim penulis.

---

## Shetty dkk. (2022)

**Sitasi APA 7:**
Shetty, B., Fernandes, R., Rodrigues, A. P., Chengoden, R., Bhattacharya, S., & Lakshmanna, K. (2022). Skin lesion classification of dermoscopic images using machine learning and convolutional neural network. *Scientific Reports, 12*, Article 18134. https://doi.org/10.1038/s41598-022-22644-9

- Akses terbuka (CC BY 4.0). Diterima 14 Juli 2022, disetujui 18 Oktober 2022. Ada catatan "corrected publication 2022" (hlm. 11), jadi pernah ada koreksi.
- Afiliasi: Government Polytechnic for Women Mangaluru, NMAM Institute of Technology, VIT Vellore.
- Data publik (Kaggle "skin-cancer-mnist-ham10000"). **Kode tidak dibuka.** Teks menyebut "the figures are visible in the code" tetapi tidak ada tautan.

### Data (hlm. 1–3)
- HAM10000: 10.015 citra JPEG 600×450 dengan metadata usia, jenis kelamin, `lesion_id`, `image_id`, `dx_type`, lokasi, dan `dx`. *Ground truth* berasal dari histopatologi (>50%), confocal, follow-up, atau konsensus.
- Jumlah per kelas (benar, sesuai Tschandl dkk., 2018): akiec 327, bcc 514, bkl 1.099, df 115, mel 1.113, **nv 6.705**, vasc 142.

### Metode (hlm. 4–7; Gbr. 2)
1. **Subset seimbang:** hanya **100 citra per kelas → 700 citra** (≈7% dari HAM10000), karena kelas terkecil (df) hanya 115 citra. Ini disebut kontribusi "lebih efisien komputasi" dibanding studi yang memakai seluruh data lalu meng-*oversample* df sekitar 50×.
2. **Resize:** 220×220 untuk ML; **96×96×3** untuk CNN, dinormalisasi ke 0–1, label diubah ke *one-hot* (`LabelBinarizer`).
3. **Augmentasi:**
   - ML: **horizontal flip saja**, 100 → 200 per kelas (**1.400 citra**).
   - CNN: transformasi acak (rotasi, *shear*, dll.) saat pelatihan.
4. **Ekstraksi fitur global untuk ML** (Gbr. 2): konversi RGB → grayscale/HSV, lalu
   - **Color Histogram** (warna)
   - **Hu Moments** (bentuk)
   - **Haralick Texture** (tekstur, GLCM)

   Ketiganya digabung menjadi satu vektor fitur dan disimpan ke HDF5. Alasan penulis: warna, bentuk, dan tekstur adalah sifat dominan di area lesi.
5. **Split:** 80:20. Tanpa augmentasi 560 latih / 140 uji; dengan augmentasi 1.120 / 280. Bagian kontribusi juga menyebut **10-fold cross-validation**, dan Tabel 5 disebut sebagai rata-rata K-fold.
6. **7 klasifier ML** (Tabel 1): LR (random_state=9), LDA (svd), KNN (k=5), DT ("Estimators=100"), RF (200 pohon), GaussianNB, SVM linear (C=1).
7. **CNN kustom** (Tabel 2), memakai Keras dan Python 3.7.9:
   - [Conv32 → BN → MaxPool 3×3 → Dropout 0,25]
   - [Conv64 → Conv64 → BN → MaxPool 2×2 → Dropout 0,25]
   - [Conv128 → BN → Conv128 → BN → MaxPool 2×2 → Dropout 0,25]
   - Flatten → Dense 1024 + BN → Dropout 0,5 → Dense 7 softmax
   - Hiperparameter (Tabel 3): Adam, *categorical cross-entropy*, **150 epoch**, batch 32, learning rate 0,001 → 0,00001.
8. Perangkat keras: laptop Intel i5 dengan RAM 8 GB, tanpa GPU yang disebutkan.

### Hasil yang diklaim (hlm. 7–9)
Akurasi ML (Tabel 4):

| | LR | LDA | KNN | DT | RF | NB | SVM |
|---|---|---|---|---|---|---|---|
| Tanpa augmentasi (700) | 48,93 | 35,18 | 41,79 | 44,29 | **58,57** | 34,11 | 42,14 |
| Flip horizontal (1.400) | 58,13 | 57,59 | 48,39 | 68,66 | **87,32** | 36,34 | 53,13 |

Perbandingan model (Tabel 5, rata-rata tertimbang):

| Metrik | CNN | RF | DT | LR | LDA | SVM | KNN | NB |
|---|---|---|---|---|---|---|---|---|
| Akurasi | 94 | 87 | 68 | 58 | 57 | 53 | 48 | 36 |
| Presisi | 88 | 94 | 75 | 56 | 62 | 54 | 54 | 49 |
| Recall | 85 | 94 | 74 | 55 | 58 | 50 | 50 | 36 |
| F1 | 86 | 94 | 74 | 55 | 54 | 50 | 50 | 35 |

- **CNN (Tabel 6): akurasi latih 95,18%, akurasi uji 86,43%.** Loss latih 0,1483, loss uji 0,4690. Penulis menyebut selisih 9% sebagai "minimal overfitting".
- Laporan per kelas CNN (Tabel 7, n uji = 280): presisi terendah pada bcc (0,55); recall terendah pada akiec (0,68). Rata-rata makro P/R/F1 = 0,87/0,85/0,85.
- InceptionV3 (pralatih) mencapai 95,14%. CNN kustom diklaim lebih ringan komputasinya.
- Tabel 8 membandingkan **95,18%** dengan 15 studi lain di HAM10000 (80,00–93,20%), dan mengklaim paling unggul.
- Saran penulis ke depan: tuning arsitektur dan hiperparameter, model pralatih lain, segmentasi, dan klasifikasi *real-time*.

### [Analisis] Inkonsistensi & kelemahan
1. **Angka utama adalah akurasi latih.** Abstrak dan Tabel 8 memakai 95,18%, padahal itu akurasi **latih** (Tabel 6). Akurasi uji 86,43%. Membandingkan akurasi latih dengan akurasi uji studi lain tidak adil.
2. **Tiga angka akurasi CNN berbeda.** Tabel 5 menyebut 94%, Tabel 6 menyebut 86,43%, dan akurasi yang dihitung dari Tabel 7 (Σ recall×support / 280) ≈ **85,0%**.
3. **Tabel 5 mustahil secara matematis.** Pada klasifikasi multikelas satu-label, *recall rata-rata tertimbang selalu sama dengan akurasi*. Tabel 5 justru menulis RF akurasi 87 tetapi recall 94, CNN akurasi 94 tetapi recall 85, LDA 57 vs 58, dan KNN 48 vs 50. Kemungkinan angka berasal dari *run*/fold berbeda atau tercampur.
4. **Kebocoran data akibat augmentasi sebelum split.** Untuk ML, 1.400 citra (asli + hasil flip) dibagi 80:20 setelah augmentasi. Akibatnya pasangan citra asli dan citra flip-nya bisa berada di set latih dan uji sekaligus. Ini menjelaskan lonjakan RF dari 58,6% ke 87,3% hanya dengan flip horizontal.
5. **Split per citra, bukan per lesi.** `lesion_id` disebut di paper tetapi tidak dipakai untuk split, padahal satu lesi bisa punya beberapa citra.
6. **Protokol evaluasi tidak jelas.** Ada klaim 10-fold CV, tetapi juga split 80:20 tetap. Tidak jelas angka mana yang berasal dari CV, dan simpangan baku antar-fold tidak dilaporkan.
7. **Subset seimbang tidak mewakili distribusi klinis.** Hanya 100 dari 6.705 nv yang dipakai. Data uji seimbang (37–44 per kelas) tidak mencerminkan prevalensi asli, sehingga akurasinya tidak bisa dibandingkan langsung dengan studi yang memakai distribusi asli.
8. **Pelatihan tidak stabil.** Kurva Gbr. 5 menunjukkan val_loss melonjak berkali-kali (puncak ±15 sekitar epoch 70). Itu tanda overfitting dan pelatihan yang tidak stabil, bukan "low loss".
9. **Detail kecil yang keliru.** "Estimators = 100" bukan parameter Decision Tree. Ada juga "OpenCV application was used to validate the ML models" dan "Dropout 0.25 Neurons". Parameter fitur (jumlah *bin* histogram, ruang warna, jarak/arah GLCM) tidak dilaporkan.

### [Analisis] Nilai bagi mata kuliah Pengolahan Citra
- Pipeline fitur klasiknya (histogram warna HSV + Hu moments + Haralick/GLCM) sangat cocok sebagai **baseline pengolahan citra** yang bisa diimplementasikan dan dijelaskan, lalu dibandingkan dengan CNN.
- Kelemahan evaluasinya memberi peluang perbaikan yang jelas: split per lesi, augmentasi hanya di data latih, CV berstrata dengan simpangan baku, data uji berdistribusi asli, metrik makro, serta praproses citra (hair removal/DullRazor, color constancy, segmentasi lesi sebelum ekstraksi fitur).

---

## Lilhore dkk. (2024)

**Sitasi APA 7:**
Lilhore, U. K., Simaiya, S., Sharma, Y. K., Kaswan, K. S., Rao, K. B. V. B., Rao, V. V. R. M., Baliyan, A., Bijalwan, A., & Alroobaea, R. (2024). A precise model for skin cancer diagnosis using hybrid U-Net and improved MobileNet-V3 with hyperparameters optimization. *Scientific Reports, 14*, Article 4299. https://doi.org/10.1038/s41598-024-54212-8

- Akses terbuka (CC BY 4.0). Diterima 29 September 2023, disetujui 9 Februari 2024.
- Data: "dapat diperoleh dengan menghubungi penulis korespondensi". **Tidak ada kode publik.**

### Usulan
- **Hybrid Standard U-Net (segmentasi) + Improved MobileNet-V3 (encoder/ekstraktor fitur)**, dengan **Bayesian optimization** (+ grid search) untuk hiperparameter dan *transfer learning* dari ImageNet (hlm. 7–10).
- Modifikasi MobileNet-V3 (hlm. 8):
  - *cross-entropy* diganti **bias loss**
  - blok **SE diganti ECA** (*efficient channel attention*) untuk mengurangi parameter
  - **koneksi lintas-lapisan** antar modul Mobile
  - **dilated convolution** untuk memperluas *receptive field*
  - versi "lebih dalam & lebih lebar" (varian large ±157 lapisan)
  - dropout, weight decay L2, batch norm, layer norm
- U-Net: Conv 3×3 → BN → ReLU (×2), up-sampling 2×2, skip connection (Gbr. 7; Tabel 5). Teks juga menyebut 4 pola konfigurasi U-Net (dengan GeLU, backbone VGG-19/DenseNet-201), tetapi tidak jelas mana yang dipakai.
- Langkah Bayesian optimization dijelaskan generik (GP surrogate, fungsi akuisisi EI/PI/UCB). **Ruang pencarian dan hasil hiperparameter optimal tidak dilaporkan.**

### Data & praproses (hlm. 4–7)
- HAM10000: 10.015 citra dermoskopi 450×600, 7 kelas. Jenis kelamin pasien 54% laki-laki, 45,5% perempuan, 0,5% tidak diketahui.
- Augmentasi untuk menyeimbangkan kelas: rotasi 25°, geser 15%, *shear* 15%, flip H/V, zoom 0,4, kecerahan, **CLAHE clip limit 4,0**. Citra di-*resize* ke 192×256 (teks), tetapi input model 224×224 (Tabel 4, Algoritma 1).
- Re-sampling (Tabel 3): total 10.015 → **19.000**.
- Split 70:15:15 (latih:validasi:uji). Pengacakan "YES".
- Hiperparameter (Tabel 4): batch 8, 100 epoch, Adam, **learning rate 0,2**, dropout 0,1, "growth rate 24", loss *categorical cross-entropy*.

### Hasil yang diklaim
Uji (Tabel 6, 100 epoch):

| Model | Presisi | Recall | F1 | Akurasi | ROC-AUC |
|---|---|---|---|---|---|
| MobileNet | 91,25 | 89,18 | 88,34 | 92,45 | 96,25 |
| ResNet-152v2 | 90,34 | 88,57 | 87,65 | 92,57 | 95,32 |
| VGG-16 | 90,78 | 89,27 | 88,95 | 92,89 | 94,21 |
| MobileNet-V2 | 92,36 | 90,78 | 89,36 | 94,47 | 91,78 |
| VGG-19 | 93,65 | 91,47 | 90,65 | 94,89 | 93,54 |
| **Usulan** | **97,84** | **95,27** | **97,32** | **98,86** | **98,45** |

- Validasi (Tabel 7): usulan memperoleh akurasi 98,03%, F1 97,03%, ROC-AUC 97,09%.
- Ablasi (Tabel 8, akurasi tanpa → dengan): U-Net 85,25 → 92,4; MobileNet-V3 88,32 → 94,38; augmentasi 86,38 → 92,38; BN 87,37 → 93,25; hiperparameter default → optimal 89,37 → 94,39.
- Keterbatasan menurut penulis: beban komputasi dua arsitektur, interpretabilitas rendah, dan perlu validasi pada data/klinik lain. Saran ke depan: kompresi model, XAI, data klinis tambahan.

### [Analisis] Inkonsistensi & kelemahan yang perlu dicatat
Poin-poin ini penting untuk bagian "analisis kritis" dan menjadi dasar desain eksperimen kita.

1. **Label kelas salah.** Tabel 3 menyebut kelas-0 "Melanoma" = 6.715 citra. Gbr. 2 di paper yang sama menunjukkan kelas terbesar adalah *melanocytic nevi* (~6.700). Distribusi resmi HAM10000 adalah nv 6.705, mel 1.113, bkl 1.099, bcc 514, akiec 327, vasc 142, df 115 (Tschandl dkk., 2018). Angka per kelas di Tabel 3 (1.000/1.000/500/250/300/235) juga tidak cocok dengan distribusi resmi, meski totalnya tetap 10.015.
2. **Skala confusion matrix tidak cocok dengan split.** Gbr. 8 (dibaca ulang dari PDF resolusi tinggi) berjumlah **11.734 sampel**, ±1.630–1.755 per kelas, jadi hampir seimbang sempurna. Padahal 15% dari 19.000 = 2.850, dan 15% dari 10.015 ≈ 1.502. Data uji tampaknya **diambil dari data hasil augmentasi/oversampling**. Ini indikasi kuat **kebocoran data** (salinan augmentasi citra yang sama ada di latih dan uji) dan evaluasi pada distribusi buatan yang seimbang, bukan distribusi klinis asli.
3. **Angka tidak konsisten antar bagian.** Akurasi yang dihitung ulang dari Gbr. 8 = 11.563/11.734 = **98,54%** (presisi/recall/F1 makro ≈ 98,5%), bukan 98,86% seperti di Tabel 6. Recall: abstrak 96,35% vs Tabel 6 95,27%. "Spesifisitas 97,32%" di abstrak sama persis dengan F1 di Tabel 6, padahal spesifisitas tidak pernah dilaporkan di tabel mana pun. Hlm. 15 menyebut "validation results with an accuracy of 98.86%" padahal itu angka uji. Hlm. 20 menyebut "accuracy rate of 98.45%" padahal itu ROC-AUC.
4. **Ablasi tidak cocok dengan hasil utama.** Konfigurasi terbaik di ablasi hanya ~94,4%, sedangkan model penuh diklaim 98,86%.
5. **Rumus metrik ditulis keliru.** Persamaan (9) akurasi memakai penyebut (TP+FN)+(TP+FP). Persamaan (6)–(7) ditulis TP(TP+FP), seharusnya TP/(TP+FP).
6. **Reprodusibilitas rendah.** Learning rate 0,2 dengan Adam tidak lazim. Tidak dijelaskan bagaimana *mask* segmentasi U-Net dilatih, karena HAM10000 tidak menyertakan *mask* resmi (perlu ISIC 2018 Task 1). Ruang pencarian Bayesian optimization tidak dilaporkan, kode tidak dibuka, dan data "on request".
7. **Augmentasi geometri ditulis sebagai matriks yang keliru.** Persamaan (1) "flip" sebenarnya matriks rotasi 3D terhadap sumbu x, dan Persamaan (2) rotasi memiliki elemen sinθ di baris ketiga.
8. **Split tingkat citra, bukan tingkat lesi/pasien.** HAM10000 berisi banyak citra dari lesi yang sama (`lesion_id`). Split acak per citra tanpa pengelompokan berpotensi bocor, dan paper tidak membahasnya.

---

## [Analisis] Perbandingan kedua referensi

| Aspek | Shetty dkk. (2022) | Lilhore dkk. (2024) |
|---|---|---|
| Pendekatan | Fitur global klasik + 7 ML; CNN kustom 3 blok | Hybrid U-Net (segmentasi) + Improved MobileNet-V3 (transfer learning) |
| Data dipakai | Subset 700 (100/kelas), augmentasi → 1.400 | 10.015 → oversampling → 19.000 |
| Resolusi input | 220×220 (ML), 96×96 (CNN) | 224×224 (teks juga menyebut 192×256) |
| Split | 80:20 (+ klaim 10-fold) | 70:15:15 |
| Optimasi hiperparameter | Manual (uji batch 5–40) | Bayesian optimization + grid search |
| Klaim utama | 95,18% (**akurasi latih**); uji 86,43% | 98,86% akurasi uji |
| Masalah utama | Augmentasi sebelum split; angka Tabel 5 mustahil; akurasi latih dijadikan klaim | Label kelas tertukar; CM 11.734 sampel ≠ ukuran split; angka tidak konsisten |
| Kode publik | Tidak | Tidak |

Pola yang sama di keduanya: **augmentasi/oversampling dilakukan sebelum pemisahan data, split per citra (bukan per lesi), evaluasi pada distribusi buatan yang seimbang, dan angka yang tidak konsisten antar tabel.** Kesamaan inilah celah metodologis utama yang bisa ditutup oleh eksperimen kita.

## [Analisis] Pelajaran untuk eksperimen kita
- Split per lesi (`lesion_id`) **sebelum** augmentasi; augmentasi/oversampling hanya pada data latih.
- Evaluasi pada distribusi kelas asli; laporkan metrik makro (presisi, recall, F1) dan confusion matrix dengan jumlah sampel yang cocok dengan ukuran split.
- Hitung ulang metrik dari confusion matrix untuk memastikan konsistensi; laporkan ruang pencarian & hasil optimasi hiperparameter; buka kode & konfigurasi.
- Ablasi harus memakai konfigurasi yang sama dengan model utama agar angkanya bisa dibandingkan.

## Referensi pendukung yang dikutip di atas
- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data, 5*, 180161. https://doi.org/10.1038/sdata.2018.161
