# Penelusuran Literatur: Kebaruan Modul Derm-MobileNetV3

Tanggal penelusuran: 8 November 2026 · Penyusun: subagent `peneliti` (draf untuk disunting pengguna)
Konteks: usulan arsitektur **Derm-MobileNetV3**, yaitu MobileNetV3-Large pralatih ImageNet (blok asli, SE dipertahankan) ditambah tiga modul yang diinisialisasi identitas atau bergerbang nol:
- **M1**: front-end praproses yang dapat dilatih: *Shades of Gray* dengan norma Minkowski *p* yang dipelajari atau diprediksi per citra, ditambah pelemah rambut *black-hat* dengan *soft-mask* yang dipelajari.
- **M2**: cabang tekstur orde-2 (*covariance*/*bilinear pooling*) dari blok tengah, sebagai padanan Haralick/GLCM yang dapat dilatih.
- **M3**: *pooling* lesi–kulit sekitar berpanduan *mask*: di dalam lesi, di cincin sekitar, dan selisih keduanya, tanpa *crop*.

**Konvensi**
- **[Fakta]** adalah isi/klaim penulis paper. **[Analisis]** adalah interpretasi kami.
- Tingkat baca sumber:
  - **(T)** teks penuh/HTML dibuka sebagian besar (lewat alat pengambil halaman, bukan dibaca baris per baris)
  - **(A)** hanya abstrak/metadata
  - **(R)** hanya diketahui dari daftar pustaka karya lain dan belum dibuka sama sekali
- Basis yang ditelusuri: mesin pencari web dengan kata kunci yang diarahkan ke arXiv, PubMed/PMC, IEEE, CVF (CVPR/ICCV), Springer (IPMI/MICCAI), MDPI, Nature/Scientific Reports, dan repositori universitas. **Google Scholar, Scopus, dan IEEE Xplore tidak dapat diakses langsung** (beberapa halaman mengembalikan 403). Jadi putusan "belum ditemukan" berarti *belum ditemukan dalam penelusuran ini*, bukan bukti bahwa karya tersebut tidak ada.

---

## 1. Ringkasan putusan per modul

| Modul | Putusan | Karya terdekat | Bedanya dengan usulan kita | Celah yang tersisa |
|---|---|---|---|---|
| **M1a**: *Shades of Gray* dengan *p* yang dipelajari / diprediksi per citra | **SEBAGIAN** | Matematikanya identik dengan **GeM pooling** (*p* dapat dilatih) dari Radenović dkk. (2019). *Color constancy* yang dipelajari untuk dermoskopi ada dalam bentuk GAN dua tahap (DermoCC-GAN; Salvi dkk., 2022). Di dermoskopi, SoG dipakai dengan *p* tetap = 6 (Barata dkk., 2015; Gessert dkk., 2020) atau sebagai augmentasi (Galdran dkk., 2017). Di histopatologi ada normalisasi warna yang dilatih *end-to-end* (LStainNorm; BeerLaNet). | Tidak ditemukan SoG dengan *p* yang dipelajari atau diprediksi per citra yang dilatih *end-to-end* bersama klasifier dermoskopi. | Penempatan GeM/SoG yang dapat diturunkan sebagai normalisasi iluminan di input klasifier lesi kulit, plus prediksi *p* per citra dan evaluasi pada uji eksternal. |
| **M1b**: pelemah rambut *black-hat* dengan *soft-mask* yang dipelajari | **SEBAGIAN** (komponen umum ada; kombinasi dermoskopi belum ditemukan) | Lapisan morfologi yang dapat dilatih sudah ada di visi umum (Masci dkk., 2013; Franchi dkk., 2020; Hermary dkk., 2022). Di dermoskopi, *black-hat* selalu tetap: Bansal & Sundaramurthy memakai 11×11 + inpainting TELEA; Hammouda dkk. (2026) memakainya hanya untuk membuat label U-Net, dan secara eksplisit menyebut **optimasi bersama penghapusan artefak + klasifikasi sebagai pekerjaan ke depan**. | Belum ditemukan *black-hat* yang parameternya ikut dilatih oleh *loss* klasifikasi lesi. | *Soft-mask* rambut yang dapat diturunkan dan dioptimasi oleh *loss* klasifikasi, dengan inisialisasi identitas, lalu diuji per strata kerambutan. |
| **M2**: cabang tekstur orde-2 dari blok tengah | **SUDAH ADA** (konsep), **SEBAGIAN** (konfigurasi) | *Bilinear pooling* untuk lesi kulit: BILSK (Calderón dkk., 2021), DACB-Net (Ahmad dkk., 2024), **SPCB-Net dengan *cross-layer bilinear-trilinear pooling*** (Qian dkk., 2024). Ada juga *Fisher vector* atas fitur ResNet (Yu dkk., 2019). Dasar umumnya: B-CNN, *compact bilinear*, iSQRT-COV, Deep TEN. | Karya yang ada memakai *backbone* berat (ResNet/VGG/Xception), *pooling* di lapisan akhir atau antar-lapisan, dan tidak ada cabang tambahan bergerbang nol di jaringan ringan. *Covariance pooling* dengan normalisasi akar matriks belum ditemukan untuk dermoskopi. | Bukan klaim kebaruan utama. Yang tersisa adalah varian ringan (MobileNetV3, blok tengah, gerbang nol), pembingkaian sebagai "Haralick yang dapat dilatih", dan uji di bawah protokol bebas kebocoran. |
| **M3**: *pooling* lesi / cincin sekitar / selisih berpanduan *mask*, tanpa *crop* | **SEBAGIAN** | **DermaKNet** (González-Díaz, 2019): *polar pooling* atas cincin radial yang dinormalisasi dari *mask* lesi. **PLCRD** (Rahi dkk., 2026, pratinjau): *embedding* lesi = *pooling* berbobot atensi A, konteks = 1−A, *mask* hanya saat latih. Ada juga *mask-guided classification* (MB-DCNN; Xie dkk., 2020), regularisasi atensi dengan segmentasi (Yan dkk., 2019), dan fitur warna dari wilayah di sekitar lesi di era *handcrafted* (Celebi dkk., 2007). Di ultrasonografi payudara: CNN intratumoral + peritumoral (Sun dkk., 2020). | Cincin DermaKNet berada di dalam lesi, dihitung dari pusat. Konteks PLCRD adalah komplemen atensi, bukan cincin kulit sekitar, dan tanpa fitur selisih. Studi peritumoral memakai *crop* ROI, bukan *pooling* fitur. | Kombinasi spesifik belum ditemukan: tiga *pooling* (dalam *mask*, cincin luar dengan lebar tertentu, selisihnya) pada peta fitur citra penuh tanpa *crop*, di jaringan ringan, dibandingkan antara *mask* Otsu dan U-Net. |
| **Posisi terhadap MobileNetV3 lain** | — | Lilhore dkk. (2024): SE→ECA, *dilated*, *cross-layer*, *bias loss*. Cheng dkk. (2024): SE→FSCA + modul multiskala. Nguyen dkk. (2022): *soft attention* + metadata. *Triple Attention MobileNetV3* (2024, judul saja). | Semua modifikasi yang ditemukan berada pada **mekanisme atensi/arsitektur**. Tidak ada yang menambahkan modul berbasis operasi pengolahan citra (CC, morfologi rambut, tekstur orde-2, konteks kulit sekitar). | Derm-MobileNetV3 dapat diposisikan sebagai "MobileNetV3 + prior pengolahan citra yang dapat diturunkan", bukan sebagai varian atensi baru. |

---

## 2. Rincian per modul

### 2.1 M1a: *Color constancy Shades of Gray* dengan *p* yang dipelajari

**Dasar metode**
- **[Fakta]** Finlayson & Trezzi (2004) (A) menunjukkan bahwa Grey-World dan Max-RGB adalah dua kasus khusus norma Minkowski. Norma L6 memberi hasil terbaik secara keseluruhan pada data kalibrasi mereka, sehingga *p* = 6 menjadi nilai baku.
- **[Fakta]** Di dermoskopi, SoG dipakai sebagai praproses **tetap**:
  - Barata dkk. (2015) (A) membandingkan Gray World, max-RGB, SoG, dan General Gray World pada sistem *bag-of-features* dengan histogram RGB 1-D. Sensitivitas naik dari 71,0% menjadi 79,7% dan spesifisitas dari 55,2% menjadi 76%.
  - Gessert dkk. (2020) (T, satu kalimat dicek) memakai "Shades of Gray color constancy method with Minkowski norm p = 6" mengikuti juara ISIC tahun sebelumnya, karena dataset latihnya sangat berbeda.
  - **[Analisis]** Catatan untuk `riset/PROPOSAL.md` H2b: Barata dkk. (2015) **tidak** memakai CNN. Frasa "konfirmasi temuan Barata dkk. (2015) pada CNN" sebaiknya diubah menjadi "menguji apakah temuan Barata dkk. (2015) pada fitur *bag-of-features* juga berlaku untuk CNN".
- **[Fakta]** Galdran dkk. (2017) (A) memakai SoG untuk menormalkan seluruh data latih, lalu menyimpan iluminan hasil estimasi dan menerapkan iluminan acak dari distribusi itu sebagai **augmentasi** (ISIC 2017). Nilai *p* tidak disebut di abstrak.

**Karya terdekat dengan "dipelajari"**
1. **GeM pooling** (Radenović dkk., 2019) (A).
   - [Fakta] Lapisan *generalized-mean pooling* dengan parameter *p* yang dapat dilatih. *p* bisa tunggal atau satu per kanal. *Max pooling* dan *average pooling* adalah kasus khususnya.
   - **[Analisis] Ini temuan paling penting untuk M1a.** Estimasi iluminan SoG per kanal adalah e_c = (mean_x I_c(x)^p)^{1/p}, yaitu **GeM pooling yang diterapkan pada piksel input**. Jadi "SoG dengan *p* yang dipelajari" secara matematis sama dengan GeM pada kanal RGB, diikuti penskalaan von Kries (I_c / e_c). Klaim "SoG dapat dilatih" sebagai operator baru **tidak aman**. Yang mungkin baru adalah penempatan dan tujuannya: normalisasi iluminan di input klasifier dermoskopi, prediksi *p* per citra, dan inisialisasi bergerbang nol.
2. **DermoCC-GAN** (Salvi dkk., 2022) (A).
   - [Fakta] *Color constancy* dirumuskan sebagai translasi citra-ke-citra dengan GAN. GAN dilatih pada keluaran algoritma heuristik kustom. Penulis mengklaim lebih baik dari metode CC standar pada *normalized median intensity*, dengan akurasi klasifikasi 7 kelas 79,2% dan Dice segmentasi 90,9%.
   - [Fakta] Tindak lanjutnya, Branciforti dkk. (2023) (A), menguji persepsi tiga dermatolog terhadap citra hasil DermoCC-GAN.
   - [Analisis] Pendekatan ini **dua tahap** (normalisasi dulu, lalu klasifier) dan tidak dioptimasi oleh *loss* klasifikasi.
3. **Normalisasi warna yang dilatih *end-to-end* di histopatologi** (A): LStainNorm (Ke dkk., 2025) dan BeerLaNet (Xu dkk., 2025, MICCAI; arXiv:2510.06592).
   - [Fakta] Keduanya diklaim dapat dipasang di depan *backbone* dan dilatih bersama tugas hilir.
   - [Analisis] Ini **preseden kuat di citra medis umum** untuk "normalisasi warna yang dipelajari bersama klasifier". Ini melemahkan klaim kebaruan konsep "front-end warna *end-to-end*", tetapi tidak untuk konteks dermoskopi dan model fisik SoG/von Kries.

**Putusan M1a: SEBAGIAN.**
- Celahnya: dalam penelusuran ini belum ditemukan klasifier dermoskopi yang menyertakan normalisasi iluminan SoG/GeM yang dapat diturunkan, dengan *p* yang dipelajari (global atau diprediksi per citra), dan dilatih bersama *loss* klasifikasi.
- **[Analisis] Risiko yang perlu diuji:**
  - Normalisasi *batch*/lapisan pertama jaringan dapat "membatalkan" efek penskalaan per kanal.
  - CC yang terlalu kuat dapat menghapus ciri warna diagnostik, misalnya *blue-white veil*. Ini sejalan dengan H2c di proposal.
  - *p* yang dipelajari bisa konvergen kembali ke nilai sekitar 6 atau ke nilai ekstrem. Nilai akhir *p* perlu dilaporkan.

### 2.2 M1b: Pelemah rambut *black-hat* dengan *soft-mask* yang dipelajari

**Praktik di dermoskopi (semuanya tetap)**
- [Fakta] Bansal & Sundaramurthy (T): *black-hat* *grayscale* dengan elemen penstruktur silang 11×11, ambang, lalu inpainting TELEA, sebelum CNN pralatih (HAM10000, rambut sintetis pada 5.000 citra). Penulis menyebut inpainting berbasis *deep learning* sebagai pekerjaan ke depan. Nilai ambang tidak disebutkan.
- [Fakta] Hammouda dkk. (2026) (T):
  - *Black-hat/bottom-hat* OpenCV dipakai **hanya untuk membuat label** *mask* artefak (dikoreksi dermatolog), lalu U-Net dilatih dengan Dice + BCE.
  - Klasifier melanoma terpisah dan tetap. Penulis **menyebut optimasi bersama penghapusan artefak dan klasifikasi sebagai pekerjaan ke depan**.
  - [Analisis] Angka di paper ini tidak konsisten antar bagian (mis. PSNR/IoU di Tabel 5 berbeda dengan teks) dan klasifiernya tidak dinamai.
- [Fakta] Hair removal berbasis *deep learning* lain (ChimeraNet, Deep Residual U-Net + TELEA, VAE/GAN) ditemukan dalam hasil pencarian, tetapi semuanya modul terpisah dari klasifier (A, tidak dibuka lebih lanjut).

**Morfologi yang dapat dilatih (visi umum)**
- [Fakta] Masci dkk. (2013) (A): kerangka belajar operator morfologi dengan *counter-harmonic mean*. Elemen penstruktur dan komposisi operator (termasuk *top-hat*) dipelajari dengan SGD.
- [Fakta] Franchi dkk. (2020) (A): lapisan morfologi dalam jaringan *end-to-end*, saling melengkapi dengan konvolusi.
- [Fakta] Hermary dkk. (2022) (A): lapisan morfologi halus (LMorph/SMorph) untuk *grayscale*, karena min/max tidak dapat diturunkan.

**Putusan M1b: SEBAGIAN.**
- Komponen "morfologi yang dapat dilatih" sudah ada.
- Celahnya: belum ditemukan dalam penelusuran ini *black-hat* yang dapat diturunkan dengan *soft-mask* rambut di dalam klasifier dermoskopi dan dioptimasi oleh *loss* klasifikasi. Pernyataan Hammouda dkk. (2026) bahwa optimasi bersama masih "pekerjaan ke depan" menjadi bukti tertulis bahwa celah ini diakui.
- **[Analisis] Catatan desain:**
  - Bila hanya ukuran elemen penstruktur dan ambang *soft-mask* yang dilatih (sigmoid dengan suhu), kebaruannya kecil tetapi mudah dijelaskan.
  - Dengan inisialisasi identitas, gerbang bisa tetap mendekati nol bila rambut tidak merugikan klasifikasi. Itu sendiri temuan yang sah dan sejalan dengan H2a. Nilai gerbang perlu dilaporkan.
  - Tanpa inpainting, "pelemahan" rambut (mis. interpolasi ke hasil *closing*) lebih mudah dibuat dapat diturunkan.

### 2.3 M2: Cabang tekstur orde-2 (*covariance/bilinear*)

**Karya dermoskopi/lesi kulit**
1. **BILSK** (Calderón dkk., 2021) (R/A).
   - [Fakta] Bilinear CNN ResNet50 × VGG16 untuk 7 kelas HAM10000. Klaim akurasi 0,9321 dan AUC 0,9810.
   - Metadata diambil dari daftar pustaka DACB-Net dan cuplikan abstrak. Paper aslinya belum dibuka.
2. **DACB-Net** (Ahmad dkk., 2024, pratinjau arXiv) (T).
   - [Fakta] *Compact bilinear pooling* antara aliran ResNet-50 dan Xception, ditambah modul atensi ganda dan cabang lokal hasil *crop*.
   - [Fakta] Akurasi HAM10000 92,86% dan ISIC 2019 94,24%. *Balanced accuracy* tidak dilaporkan.
   - [Analisis] Pembagian data HAM10000 tidak konsisten antar bagian (70/15/15 vs 80/20).
3. **Bilinear CNN + Constrained Triplet Loss** (Ahmad dkk., 2024, pratinjau arXiv) (A): klaim rerata akurasi 93,72%.
4. **SPCB-Net** (Qian dkk., 2024, IEEE Access) (A).
   - [Fakta] *Self-interactive attention pyramid* + ***cross-layer bilinear-trilinear pooling***. Klaim 97,10% di HAM10000 (ResNet101).
   - [Analisis] **Ini yang paling dekat dengan "orde-2 dari blok tengah"**, karena *pooling*-nya lintas lapisan. Daftar penulis diambil dari agregator dan belum dicek di laman penerbit.
5. **Yu dkk. (2019)**, IEEE TBME (A): fitur ResNet pralatih diagregasi dengan *Fisher vector* (statistik orde-1 dan orde-2 relatif terhadap GMM), lalu SVM. Ini statistik orde lebih tinggi untuk melanoma, tetapi tidak *end-to-end*.
6. Disebut di daftar pustaka DACB-Net, belum dibuka (R): Pundir dkk. (2023, ICESC) *bilinear CNN + soft attention*, dan Wei & Ji (2024) *bilinear fusion* multimodal.

**Dasar umum** (A): B-CNN (Lin dkk., 2015), *compact bilinear pooling* (Gao dkk., 2016), iSQRT-COV (Li dkk., 2018), Deep TEN (Zhang dkk., 2017). [Fakta] Deep TEN adalah lapisan *encoding* tekstur yang dapat dilatih dan menggeneralisasi VLAD/*Fisher vector*.

**GLCM yang dapat diturunkan**
- [Fakta] GLCM-MAE (Madan dkk., 2025, pratinjau arXiv:2507.10869) (A) membuat GLCM "lunak" dengan estimasi kernel densitas dan memakainya sebagai *loss* rekonstruksi pada pralatih MAE citra medis, bukan sebagai cabang klasifier.
- [Analisis] Ini alternatif yang lebih "Haralick-setia" daripada *covariance pooling*, bila ingin padanan yang lebih harfiah.

**Putusan M2: SUDAH ADA (konsep) / SEBAGIAN (konfigurasi).**
- *Bilinear pooling* untuk klasifikasi lesi kulit sudah ada sejak 2021, termasuk versi lintas lapisan (SPCB-Net).
- Yang belum ditemukan:
  - (a) cabang orde-2 pada jaringan ringan MobileNetV3 dari blok tengah, sebagai cabang tambahan bergerbang nol di atas *backbone* yang dipertahankan;
  - (b) *covariance pooling* dengan normalisasi akar matriks (iSQRT-COV) untuk dermoskopi;
  - (c) evaluasi dengan protokol bebas kebocoran dan BACC. Semua karya di atas melaporkan akurasi, sebagian dengan pembagian yang tidak jelas.
- **[Analisis]** Jangan mengklaim M2 sebagai kebaruan metode. Posisikan sebagai "padanan GLCM/Haralick yang dapat dilatih, diuji ablasi terhadap Haralick *handcrafted* pada protokol yang sama". Ini terhubung langsung ke RQ3 proposal.

### 2.4 M3: *Pooling* lesi–kulit sekitar berpanduan *mask*

**Karya dermoskopi**
1. **DermaKNet** (González-Díaz, 2019, IEEE JBHI). Dibaca dari versi pratinjau arXiv:1703.01976 (T); versi jurnalnya belum dibuka.
   - [Fakta] Dari *mask* lesi dihitung *Normalized Polar Coordinates*: *mask* didekati elips dengan momen orde-2 yang sama, lalu dipetakan ke lingkaran satuan.
   - [Fakta] Blok ***polar pooling*** melakukan *average/max pooling* atas sektor polar (R cincin × A sudut; 3×6 pada konfigurasi mereka). Batas cincin dibuat tidak seragam agar jumlah pikselnya sama.
   - [Fakta] Ada pula blok asimetri (selisih antar-sektor yang dilipat) dan blok modulasi oleh peta struktur dermoskopik.
   - **[Analisis]** Ini preseden terkuat untuk "*pooling* cincin berpanduan *mask*". Bedanya: cincin DermaKNet berpusat di tengah lesi dan dinormalisasi ke radius lesi (radius 0–1), tidak secara eksplisit mencakup **pita kulit di luar batas lesi**, dan tidak menghitung selisih lesi−kulit sekitar. Blok asimetrinya adalah selisih antar-sektor, bukan dalam−luar. **Wajib dikutip** agar klaim M3 tidak dianggap mengabaikan prior art.
2. **PLCRD** (Rahi dkk., 2026, pratinjau arXiv:2607.18773, belum terbit jurnal) (T).
   - [Fakta] *Teacher* (Swin-Tiny) menerima citra dan citra-lesi-ber-*mask*, sedangkan *student* (EfficientFormerV2-S0) hanya menerima citra.
   - [Fakta] *Embedding* lesi z^l = Pool(F, A) dan konteks z^b = Pool(F, 1−A), dengan A peta atensi yang dikondisikan kelas. Relasi lesi–konteks didistilasi (*lesion-context affinity matching*).
   - [Fakta] HAM10000 dibagi **per lesi** dengan uji eksternal ISIC 2018. Klaim BACC 0,764 ± 0,023, F1 makro 0,773, dan F1 makro ISIC 2018 0,732. MobileNetV3-Small hanya muncul sebagai pembanding.
   - **[Analisis]** Ini sangat dekat secara gagasan (lesi vs konteks) dan protokolnya bersih. Bedanya: konteks PLCRD adalah komplemen atensi yang dipelajari (seluruh non-lesi), bukan cincin kulit sekitar dengan lebar terkontrol. Tidak ada fitur selisih eksplisit. *Mask* hanya dipakai saat latih, lewat distilasi. Angka BACC-nya berguna sebagai **pembanding realistis** untuk HAM10000 dengan protokol per lesi.
3. **MB-DCNN** (Xie dkk., 2020, IEEE TMI) (A).
   - [Fakta] *Coarse segmentation network* menghasilkan *mask* sebagai prior untuk *mask-guided classification network*, yang peta lokalisasinya lalu membantu segmentasi lanjutan. ISIC 2017 dan PH2, rerata AUC 93,8% / 97,7%.
   - [Analisis] *Mask* dipakai sebagai prior/atensi, bukan *pooling* dalam/luar.
4. **Yan dkk. (2019)**, IPMI (A): modul atensi yang dapat dilatih *end-to-end*, dengan regularisasi peta atensi memakai segmentasi lesi atau peta fitur dermoskopik (ISIC 2016/2017).
5. **Celebi dkk. (2007)** (A).
   - [Fakta] Setelah deteksi batas lesi, citra dibagi menjadi beberapa "wilayah yang bermakna klinis" dengan *Euclidean distance transform*, lalu fitur warna/tekstur diekstraksi per wilayah (564 citra, spesifisitas 92,34%, sensitivitas 93,33%).
   - [PERLU VERIFIKASI teks penuh] Definisi persisnya (pinggiran dalam/luar lesi, kulit latar, fitur warna relatif lesi−kulit) belum terbaca karena PDF tidak dapat diakses. Bila terkonfirmasi, ini preseden *handcrafted* langsung untuk ide "dalam, cincin, dan selisih".
6. **Bukti tentang latar dan *crop***
   - [Fakta] Mahbod dkk. (2020) (A): pada ISIC 2017, *mask* tidak signifikan membantu melanoma, *crop* dengan *mask* yang diperlebar membantu SK, dan **menghapus seluruh latar menurunkan kinerja**.
   - [Fakta] Bissoto dkk. (2019) (A): model masih dapat mengklasifikasikan lesi dengan cukup baik **meski informasi lesi dihapus**. Ini tanda korelasi palsu dari latar/artefak.
   - [Analisis] Keduanya mendukung keputusan "tanpa *crop*, latar tidak dibuang". Bissoto dkk. sekaligus menjadi **ancaman validitas**: cincin kulit sekitar bisa menjadi jalan pintas (*shortcut*) melalui artefak seperti penggaris, sudut gelap, atau tinta. Perlu ablasi "cincin saja" dan "lesi saja", serta analisis per sumber data.

**Citra medis umum (peritumoral)**
- [Fakta] Sun dkk. (2020) (T): pada USG payudara, CNN DenseNet-121 dilatih pada ROI intratumoral, peritumoral (kontur didilatasi sekitar 5 mm), dan gabungan. AUC uji: gabungan 0,912, peritumoral 0,775, intratumoral 0,748.
- [Analisis] Paradigma peritumoral sudah mapan di radiologi/radiomik, tetapi dilakukan dengan *crop* ROI terpisah, bukan *pooling* fitur pada peta fitur citra penuh.

**Putusan M3: SEBAGIAN.**
- Komponen-komponennya sudah ada: *pooling* cincin dari *mask* (DermaKNet), lesi vs konteks (PLCRD), *mask* sebagai atensi (MB-DCNN, Yan dkk.), dan region peritumoral (Sun dkk.).
- Yang belum ditemukan dalam penelusuran ini adalah kombinasi berikut:
  - (a) tiga vektor (*pooling* dalam *mask*, *pooling* pita kulit di luar batas dengan lebar tertentu, dan selisih keduanya) pada peta fitur **citra penuh tanpa *crop***;
  - (b) dipasang bergerbang nol di MobileNetV3 pralatih;
  - (c) dibandingkan dengan sumber *mask* Otsu vs U-Net dan dievaluasi dengan protokol per lesi plus uji eksternal.

---

## 3. Posisi Derm-MobileNetV3 terhadap MobileNetV3 termodifikasi lain

| Karya | Modifikasi | Data & protokol | Catatan [Analisis] |
|---|---|---|---|
| Howard dkk. (2019) (A) | MobileNetV3 asli: NAS + NetAdapt, SE, h-swish | ImageNet | Dasar arsitektur. |
| Lilhore dkk. (2024) (T, referensi utama) | SE→ECA, *dilated conv*, *cross-layer connection*, *bias loss*, + U-Net | HAM10000 di-*oversample* menjadi 19.000, split 70:15:15 per citra | Indikasi kebocoran (lihat `referensi/RINGKASAN-REFERENSI-UTAMA.md`). |
| Cheng dkk. (2024) (T) | SE pada MobileNetV3 diganti FSCA (*fused spatial-channel attention*) + modul fitur multiskala | ISIC 2019 (8 kelas), split 80/20 tanpa set validasi, tidak disebut per lesi | Akurasi 87,53%, AUC 0,9631. BACC tidak dilaporkan. Praproses hanya *resize*. |
| Nguyen dkk. (2022) (T) | Dua konv 1×1 terakhir diganti *soft attention*, + metadata, + *loss* tertimbang baru | HAM10000, split acak 90/10 (tidak disebut per lesi), uji 857 citra | MobileNetV3-Large: akurasi 0,86, BACC 0,87 (Tabel A5). Ada inkonsistensi jumlah augmentasi dan AUC antar-tabel. |
| *Triple Attention MobileNetV3* (2024, IEEE COMPAS, doi:10.1109/COMPAS60761.2024.10796988) (hanya judul/DOI) | Tiga modul atensi (dari judul) | Tidak diketahui | Isi belum dibaca. [PERLU SUMBER: penulis dan isi] |
| MobileNetV3 + CBAM untuk ISIC 2024 SLICE-3D (jurnal JUTIF, Unsoed) (metadata belum terbaca) | CBAM + tuning Bayesian | ISIC 2024 (bukan dermoskopi klasik) | [PERLU SUMBER: metadata lengkap]. PDF ditemukan tetapi teksnya tidak dapat diekstraksi. |
| Dai dkk. (2024), HierAttn (A) | Atensi multi-tahap/multi-cabang dengan *deep supervision* | ISIC 2019, PAD-UFES-20 | MobileNetV3 hanya pembanding. *Backbone* HierAttn tidak terkonfirmasi dari abstrak. |

**[Analisis] Pola yang terlihat.** Semua modifikasi MobileNetV3 untuk lesi kulit yang ditemukan bekerja pada **mekanisme atensi** (ECA, FSCA, *soft attention*, CBAM, *triple attention*) atau koneksi arsitektur. Tidak ada yang:
1. menanamkan operasi pengolahan citra klasik (CC, morfologi rambut) sebagai lapisan yang dapat diturunkan;
2. menambah cabang statistik tekstur orde-2;
3. memakai *mask* untuk *pooling* lesi vs kulit sekitar.

Selain itu, tidak ada satu pun yang melaporkan protokol per lesi plus uji eksternal (kecuali PLCRD, yang bukan MobileNetV3). Derm-MobileNetV3 dapat diposisikan sebagai **"MobileNetV3 dengan prior pengolahan citra yang dapat diturunkan dan diinisialisasi identitas"**, berbeda arah dengan Lilhore dkk. (2024) yang **mengganti** SE dengan ECA. Kita mempertahankan SE dan menambah modul, sehingga ablasi "MobileNetV3 asli vs + M1/M2/M3" menjadi bersih.

---

## 4. Rekomendasi rumusan klaim kebaruan

**Prinsip umum [Analisis]**
1. **Hindari** kata "pertama" (*first*) dan "belum pernah". Gunakan: "Sepengetahuan kami, berdasarkan penelusuran pada [tanggal] di arXiv, PubMed/PMC, IEEE, CVF, Springer, dan MDPI dengan kata kunci [daftar], belum ditemukan karya yang …".
2. Kebaruan sebaiknya diletakkan pada **kombinasi + pembingkaian + evaluasi**, bukan pada operator individual. GeM, *bilinear pooling*, morfologi yang dapat dilatih, dan *polar pooling* sudah ada.
3. Kutip karya terdekat secara eksplisit di bagian "Karya terkait" beserta bedanya. Minimal: Radenović dkk. (2019), Salvi dkk. (2022), Hammouda dkk. (2026), Masci dkk. (2013)/Hermary dkk. (2022), Calderón dkk. (2021), Qian dkk. (2024), González-Díaz (2019), Rahi dkk. (2026), Mahbod dkk. (2020), Lilhore dkk. (2024), dan Cheng dkk. (2024).

**Contoh rumusan per modul**
- **M1a:** "Kami merumuskan *Shades of Gray* sebagai lapisan normalisasi iluminan yang dapat diturunkan. Estimasi iluminannya setara dengan *generalized-mean pooling* (Radenović dkk., 2019) pada kanal RGB, dengan eksponen *p* yang dipelajari bersama klasifier dan diinisialisasi bergerbang nol. Berbeda dengan DermoCC-GAN (Salvi dkk., 2022) yang memisahkan normalisasi dan klasifikasi, modul ini dioptimasi langsung oleh *loss* klasifikasi."
- **M1b:** "Kami menjadikan *black-hat* DullRazor-style sebagai pelemah rambut lunak yang parameternya ikut dilatih. Optimasi bersama penghapusan artefak dan klasifikasi disebut sebagai arah yang belum dikerjakan oleh Hammouda dkk. (2026)."
- **M2 (bukan klaim utama):** "Kami menguji cabang *covariance pooling* ringan dari blok tengah MobileNetV3 sebagai padanan tekstur Haralick yang dapat dilatih, dan membandingkannya secara langsung dengan fitur Haralick *handcrafted* pada protokol yang sama. *Bilinear pooling* untuk lesi kulit telah dikaji sebelumnya (Calderón dkk., 2021; Qian dkk., 2024); kontribusi kami adalah evaluasinya pada jaringan ringan di bawah protokol per lesi."
- **M3:** "Memperluas *polar pooling* berpanduan *mask* (González-Díaz, 2019) dan pemisahan lesi–konteks (Rahi dkk., 2026), kami mem-*pool* fitur di dalam lesi, di pita kulit sekitar dengan lebar terkontrol, serta selisih keduanya pada citra penuh tanpa *crop*. Ini memformalkan perbandingan lesi–kulit sekitar yang lazim dalam fitur warna *handcrafted* [PERLU VERIFIKASI Celebi dkk., 2007]."
- **Kontribusi gabungan yang paling aman:** "(i) Tiga operasi pengolahan citra klasik dermoskopi (CC, pelemahan rambut, konteks kulit sekitar) beserta padanan tekstur orde-2 dirumuskan sebagai modul MobileNetV3 yang dapat diturunkan dan diinisialisasi identitas. (ii) Ablasi per modul dilakukan di bawah protokol bebas kebocoran (split per lesi, uji ISIC 2018 resmi). (iii) Nilai parameter yang dipelajari (*p*, gerbang, lebar cincin) dianalisis sebagai bukti tahap pengolahan citra mana yang benar-benar dipakai jaringan."
- Poin (iii) adalah sudut yang paling khas. Gerbang yang tetap mendekati nol pun tetap merupakan temuan, dan ini menyambung dengan RQ2 proposal.

**Catatan keselarasan dengan proposal [Analisis].** `riset/PROPOSAL.md` (v1) adalah studi evaluasi ulang + ablasi, tanpa metode baru. Derm-MobileNetV3 menjawab kemungkinan dosen meminta "metode usulan baru" (Pertanyaan untuk dosen no. 2). Modul M1–M3 bisa dirancang sebagai **versi terlatih dari faktor RQ2/RQ4**, sehingga kedua arah tidak saling bertentangan.

---

## 5. Celah penelusuran yang belum terjawab
1. Google Scholar, Scopus, dan IEEE Xplore tidak diakses langsung. Penelusuran sitasi maju (*forward citation*) terhadap DermaKNet, BILSK, dan GeM di Google Scholar belum dilakukan dan **wajib** sebelum klaim dibekukan.
2. Teks penuh Celebi dkk. (2007), BILSK, SPCB-Net, dan versi jurnal DermaKNet belum dibaca.
3. Prosiding MICCAI/ISBI 2023–2026 belum disisir khusus untuk "*learnable color constancy*" dan "*hair-aware*". Hasil pencarian umum tidak memunculkannya, tetapi ini belum sama dengan penyisiran per prosiding.
4. Prediksi *p* per citra mirip estimasi iluminan berbasis CNN (mis. FC4, Bianco dkk.) di visi umum. Belum ditelusuri apakah ada versi "prediksi eksponen Minkowski" yang eksplisit di luar dermoskopi. [PERLU SUMBER]
5. Metadata *Triple Attention MobileNetV3* (COMPAS 2024) dan artikel JUTIF (CBAM + MobileNetV3) belum lengkap.

---

## 6. Daftar pustaka (APA 7)

Tingkat baca: (T) teks penuh/HTML dibuka sebagian besar; (A) abstrak/metadata; (R) hanya dari daftar pustaka karya lain.

- Ahmad, B., Usama, M., Ahmad, T., Saeed, A., Khatoon, S., & Chen, M. (2024). *DACB-Net: Dual attention guided compact bilinear convolution neural network for skin disease classification* (arXiv:2407.03439) [Pratinjau]. arXiv. https://arxiv.org/abs/2407.03439 (T)
- Ahmad, B., Usama, M., Ahmad, T., Saeed, A., Khatoon, S., & Hu, L. (2024). *Bilinear-convolutional neural network using a matrix similarity-based joint loss function for skin disease classification* (arXiv:2406.00696) [Pratinjau]. arXiv. https://arxiv.org/abs/2406.00696 (A)
- Bansal, N., & Sundaramurthy, S. (2023). Integrated convolutional neural network for skin cancer classification with hair and noise restoration. *Turkish Journal of Medical Sciences, 55*(1), 161–177. https://doi.org/10.55730/1300-0144.5954 (T) [tahun dan volume sesuai tampilan PMC (PMC11913500); volume 55 lazimnya tahun 2025, jadi verifikasi tahun di laman jurnal]
- Barata, C., Celebi, M. E., & Marques, J. S. (2015). Improving dermoscopy image classification using color constancy. *IEEE Journal of Biomedical and Health Informatics, 19*(3), 1146–1152. https://doi.org/10.1109/JBHI.2014.2336473 (A)
- Bissoto, A., Fornaciali, M., Valle, E., & Avila, S. (2019). (De)constructing bias on skin lesion datasets. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops* (pp. 2766–2774). https://doi.org/10.1109/CVPRW.2019.00335 (A)
- Branciforti, F., Meiburger, K. M., Zavattaro, E., Veronese, F., Tarantino, V., Mazzoletti, V., Di Cristo, N., Savoia, P., & Salvi, M. (2023). Impact of artificial intelligence-based color constancy on dermoscopical assessment of skin lesions: A comparative study. *Skin Research and Technology, 29*(11), e13508. https://doi.org/10.1111/srt.13508 (A)
- Calderón, C., Sanchez, K., Castillo, S., & Arguello, H. (2021). BILSK: A bilinear convolutional neural network approach for skin lesion classification. *Computer Methods and Programs in Biomedicine Update, 1*, 100036. [DOI belum diverifikasi] (R)
- Celebi, M. E., Kingravi, H. A., Uddin, B., Iyatomi, H., Aslandogan, Y. A., Stoecker, W. V., & Moss, R. H. (2007). A methodological approach to the classification of dermoscopy images. *Computerized Medical Imaging and Graphics, 31*(6), 362–373. https://doi.org/10.1016/j.compmedimag.2007.01.003 (A)
- Cheng, H., Lian, J., & Jiao, W. (2024). Enhanced MobileNet for skin cancer image classification with fused spatial channel attention mechanism. *Scientific Reports, 14*, Article 28850. https://doi.org/10.1038/s41598-024-80087-w (T)
- Dai, W., Liu, R., Wu, T., Wang, M., Yin, J., & Liu, J. (2024). Deeply supervised skin lesions diagnosis with stage and branch attention. *IEEE Journal of Biomedical and Health Informatics, 28*(2), 719–729. https://doi.org/10.1109/JBHI.2023.3308697 (A)
- Finlayson, G. D., & Trezzi, E. (2004). Shades of gray and colour constancy. In *Proceedings of the 12th Color Imaging Conference* (pp. 37–41). Society for Imaging Science and Technology. https://ueaeprints.uea.ac.uk/23682 (A)
- Franchi, G., Fehri, A., & Yao, A. (2020). Deep morphological networks. *Pattern Recognition, 102*, Article 107246. https://doi.org/10.1016/j.patcog.2020.107246 (A) [DOI disusun dari nomor artikel; cocokkan di laman penerbit]
- Galdran, A., Alvarez-Gila, A., Meyer, M. I., Saratxaga, C. L., Araújo, T., Garrote, E., Aresta, G., Costa, P., Mendonça, A. M., & Campilho, A. (2017). *Data-driven color augmentation techniques for deep skin image analysis* (arXiv:1703.03702) [Pratinjau]. arXiv. https://arxiv.org/abs/1703.03702 (A)
- Gao, Y., Beijbom, O., Zhang, N., & Darrell, T. (2016). Compact bilinear pooling. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 317–326). https://doi.org/10.1109/CVPR.2016.41 (A)
- Gessert, N., Nielsen, M., Shaikh, M., Werner, R., & Schlaefer, A. (2020). Skin lesion classification using ensembles of multi-resolution EfficientNets with meta data. *MethodsX, 7*, 100864. https://doi.org/10.1016/j.mex.2020.100864 (T, satu bagian)
- González-Díaz, I. (2019). DermaKNet: Incorporating the knowledge of dermatologists to convolutional neural networks for skin lesion diagnosis. *IEEE Journal of Biomedical and Health Informatics, 23*(2), 547–559. https://doi.org/10.1109/JBHI.2018.2806962 (T untuk pratinjau arXiv:1703.01976; versi jurnal belum dibuka)
- Hammouda, N. G., Shalaby, M., Alfilh, R. H. C., & Singh, N. S. S. (2026). Hair and artifact removal in dermoscopic images using deep learning for enhanced skin cancer detection. *Scientific Reports, 16*, Article 14452. https://doi.org/10.1038/s41598-026-44545-x (T)
- Hermary, R., Tochon, G., Puybareau, É., Kirszenberg, A., & Angulo, J. (2022). Learning grayscale mathematical morphology with smooth morphological layers. *Journal of Mathematical Imaging and Vision, 64*, 736–753. https://doi.org/10.1007/s10851-022-01091-1 (A)
- Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019). Searching for MobileNetV3. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 1314–1324). https://doi.org/10.1109/ICCV.2019.00140 (A) [nama penulis ke-3 s.d. ke-12 dari pengetahuan umum; cocokkan di laman CVF]
- Ke, J., dkk. (2025). Learnable color space conversion and fusion for stain normalization in pathology images (LStainNorm). *Medical Image Analysis, 101*, 103424. [daftar penulis lengkap dan DOI belum diverifikasi] (A)
- Li, P., Xie, J., Wang, Q., & Gao, Z. (2018). Towards faster training of global covariance pooling networks by iterative matrix square root normalization. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. https://arxiv.org/abs/1712.01034 (A) [halaman prosiding belum diverifikasi]
- Lilhore, U. K., Simaiya, S., Sharma, Y. K., Kaswan, K. S., Rao, K. B. V. B., Rao, V. V. R. M., Baliyan, A., Bijalwan, A., & Alroobaea, R. (2024). A precise model for skin cancer diagnosis using hybrid U-Net and improved MobileNet-V3 with hyperparameters optimization. *Scientific Reports, 14*, Article 4299. https://doi.org/10.1038/s41598-024-54212-8 (T, referensi utama proyek)
- Lin, T.-Y., RoyChowdhury, A., & Maji, S. (2015). Bilinear CNN models for fine-grained visual recognition. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 1449–1457). (R/A)
- Madan, C., dkk. (2025). *Focus on texture: Rethinking pre-training in masked autoencoders for medical image classification* (arXiv:2507.10869) [Pratinjau]. arXiv. https://arxiv.org/abs/2507.10869 (A) [daftar penulis lengkap belum diverifikasi]
- Mahbod, A., Tschandl, P., Langs, G., Ecker, R., & Ellinger, I. (2020). The effects of skin lesion segmentation on the performance of dermatoscopic image classification. *Computer Methods and Programs in Biomedicine, 197*, 105725. https://doi.org/10.1016/j.cmpb.2020.105725 (A)
- Masci, J., Angulo, J., & Schmidhuber, J. (2013). A learning framework for morphological operators using counter-harmonic mean. In *Mathematical Morphology and Its Applications to Signal and Image Processing (ISMM 2013)*, Lecture Notes in Computer Science (Vol. 7883, pp. 329–340). Springer. https://doi.org/10.1007/978-3-642-38294-9_28 (A) [nama penulis kedua/ketiga dan halaman dari pengetahuan umum; cocokkan di laman Springer]
- Nguyen, V. D., Bui, N. D., & Do, H. K. (2022). Skin lesion classification on imbalanced data using deep learning with soft attention. *Sensors, 22*(19), 7530. https://doi.org/10.3390/s22197530 (T)
- Qian, X., Weng, T., Han, Q., Wu, C., Xu, H., Hou, M., Qiu, Z., Zhou, B., & Gao, X. (2024). SPCB-Net: A multi-scale skin cancer image identification network using self-interactive attention pyramid and cross-layer bilinear-trilinear pooling. *IEEE Access*. [volume, halaman, dan DOI belum diverifikasi; penulis dari agregator] (A)
- Radenović, F., Tolias, G., & Chum, O. (2019). Fine-tuning CNN image retrieval with no human annotation. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 41*(7), 1655–1668. https://arxiv.org/abs/1711.02512 (A) [volume/halaman/DOI jurnal dari pengetahuan umum; cocokkan]
- Rahi, A. M., Hossain, M. M., Joy, M. Z. H., Mridha, M. F., & Hossen, M. J. (2026). *Privileged lesion-context relational distillation for mask-free skin lesion classification* (arXiv:2607.18773) [Pratinjau]. arXiv. https://arxiv.org/abs/2607.18773 (T)
- Salvi, M., Branciforti, F., Veronese, F., dkk. (2022). DermoCC-GAN: A new approach for standardizing dermatological images using generative adversarial networks. *Computer Methods and Programs in Biomedicine, 225*, 107040. https://doi.org/10.1016/j.cmpb.2022.107040 (A) [daftar penulis lengkap belum diverifikasi]
- Sun, Q., Lin, X., Zhao, Y., Li, L., Yan, K., Liang, D., Sun, D., & Li, Z.-C. (2020). Deep learning vs. radiomics for predicting axillary lymph node metastasis of breast cancer using ultrasound images: Don't forget the peritumoral region. *Frontiers in Oncology, 10*, 53. https://doi.org/10.3389/fonc.2020.00053 (T)
- Xie, Y., Zhang, J., Xia, Y., & Shen, C. (2020). A mutual bootstrapping model for automated skin lesion segmentation and classification. *IEEE Transactions on Medical Imaging, 39*(7), 2482–2493. https://doi.org/10.1109/TMI.2020.2972964 (A)
- Xu, T., dkk. (2025). BeerLaNet: Adaptive stain normalization for cross-domain medical histology. In *MICCAI 2025* (arXiv:2510.06592). https://arxiv.org/abs/2510.06592 (A) [judul persis dan daftar penulis belum diverifikasi]
- Yan, Y., Kawahara, J., & Hamarneh, G. (2019). Melanoma recognition via visual attention. In *Information Processing in Medical Imaging (IPMI 2019)*, Lecture Notes in Computer Science (Vol. 11492, pp. 793–804). Springer. https://doi.org/10.1007/978-3-030-20351-1_62 (A)
- Yu, Z., Jiang, X., Zhou, F., Qin, J., dkk. (2019). Melanoma recognition in dermoscopy images via aggregated deep convolutional features. *IEEE Transactions on Biomedical Engineering, 66*(4), 1006–1016. https://doi.org/10.1109/TBME.2018.2866166 (A) [daftar penulis lengkap belum diverifikasi]
- Zhang, H., Xue, J., & Dana, K. (2017). Deep TEN: Texture encoding network. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 708–717). https://doi.org/10.1109/CVPR.2017.309 (A)

**Ditemukan tetapi metadata belum lengkap (tidak dikutip sebagai rujukan)**
- *Triple Attention MobileNetV3: Harnessing integrated attention and transfer learning for next-generation skin lesion detection* (IEEE COMPAS 2024), doi:10.1109/COMPAS60761.2024.10796988. Hanya judul dan DOI dari ResearchGate.
- Artikel JUTIF (Universitas Jenderal Soedirman) tentang MobileNetV3 + CBAM untuk ISIC 2024. PDF: https://jutif.if.unsoed.ac.id/index.php/jurnal/article/download/4546/859/13953
- Pundir dkk. (2023, ICESC) dan Wei & Ji (2024, *Multimedia Tools and Applications*). Dari daftar pustaka DACB-Net, tentang *bilinear* untuk lesi kulit.
