# Plan 0 — Sidang Ide (Profesor vs Pengusul)

Format: setiap aspek diserang oleh **Profesor (P)**, dijawab oleh **Pengusul (U)**, lalu diberi **Putusan**: ✅ diterima · 🔧 diterima dengan revisi · ❓ terbuka (harus dijawab di plan berikutnya). Bagian "Perubahan" mencatat apa yang berubah di proposal.

Fakta yang dipakai sebagai amunisi diverifikasi dari sumber primer atau dihitung sendiri. Daftarnya ada di §Lampiran.

Ide yang disidangkan (versi masuk, dari `VALIDASI-IDE.md`):
> Evaluasi ulang bebas kebocoran terhadap pipeline Shetty dkk. (2022) dan Lilhore dkk. (2024) untuk mengukur kontribusi nyata tahapan pengolahan citra (RQ1 audit, RQ2 praproses, RQ3 handcrafted vs CNN vs fusi; RQ4 segmentasi opsional).

---

## Babak 1 — Kebaruan

**P1.1:** "Replikasi bukan penelitian. Anda hanya mengulang pekerjaan orang dan menunjukkan angkanya turun. Kebocoran per citra di HAM10000 sudah ditunjukkan LesionBench dan beberapa audit. Apa yang baru?"

**U:** Benar bahwa "split per lesi menurunkan angka" bukan hal baru, dan kami tidak mengklaimnya. Ada tiga kebaruan:
1. **Dekomposisi inflasi.** Kami tidak hanya menunjukkan bahwa angka turun, tetapi juga **berapa kontribusi tiap pelanggaran protokol**: augmentasi sebelum split, split per citra, uji seimbang buatan, dan pelaporan akurasi latih. Studi yang ada hanya membandingkan dua kondisi (per citra vs per lesi).
2. **Audit terhadap dua paper spesifik** yang terbit di jurnal bereputasi dan tidak punya replikasi atau kritik sama sekali (sudah dicek).
3. **Ablasi praproses per langkah dengan protokol bersih**, yang belum ada (studi yang ada menggabungkan beberapa langkah sekaligus).

**P1.2:** "Untuk poin 3: Barata, Celebi & Marques (2015, IEEE JBHI) sudah menunjukkan *color constancy* meningkatkan klasifikasi citra multisumber. Hipotesis H2 Anda sudah terjawab sepuluh tahun lalu."

**U:** Barata dkk. memakai fitur klasik (*Bag-of-Features*) dan data multisumber lama, sebelum era CNN pralatih yang punya normalisasi dan augmentasi warna sendiri. Pertanyaan yang masih terbuka adalah **apakah manfaat itu bertahan pada CNN modern** yang dilatih dengan *color jitter*. Jadi kami tidak mengklaim CC sebagai temuan baru. H2 diposisikan sebagai **uji konfirmasi lintas era**: apakah temuan Barata berlaku pada CNN, dan apakah efeknya lebih besar pada partisi eksternal ISIC 2018 (316 citra dari Turki, Selandia Baru, Swedia, Argentina)?

**Putusan:** 🔧 Diterima dengan revisi.
**Perubahan:**
- Klaim kebaruan ditulis eksplisit dalam tiga butir di atas.
- Kata "baru" untuk CC diganti menjadi "uji konfirmasi temuan Barata dkk. (2015) pada CNN modern".
- Barata 2015 masuk sumber wajib Plan 2.

---

## Babak 2 — Signifikansi: siapa yang butuh?

**P2.1:** "Dokter tidak membaca paper replikasi. Reviewer jurnal tidak suka hasil negatif. Siapa yang butuh penelitian ini?"

**U:** Ada tiga kelompok:
- **Peneliti dan reviewer.** Literatur HAM10000 dipenuhi angka >95% dari protokol yang bocor. Angka dekomposisi inflasi memberi patokan untuk menilai klaim semacam itu.
- **Pengembang sistem di fasilitas terbatas.** Mereka perlu tahu tahap praproses **mana yang layak** ditambahkan ke pipeline ringan, dan mana yang hanya menambah beban komputasi tanpa manfaat.
- **Mahasiswa dan mata kuliah.** Ini latihan metodologi yang benar, yaitu luaran tugas.

**P2.2:** "Bagaimana kalau semua efek praproses ternyata nol? Anda tidak punya hasil."

**U:** Hasil nol tetap informatif **jika dirancang untuk bisa menyimpulkan 'tidak ada efek berarti'**. Untuk itu kami menetapkan **SESOI** (*smallest effect size of interest*) di muka, lalu memakai **uji ekuivalensi (TOST)**. Jadi "efek < SESOI" adalah kesimpulan yang sah, bukan kegagalan. Contoh kalimat yang bisa ditulis: "DullRazor tidak memberi manfaat ≥2 poin BACC pada CNN; tidak perlu dipakai."

**Putusan:** 🔧 Diterima dengan revisi.
**Perubahan:**
- Tambahkan SESOI = **2 poin persentase BACC**. Alasan awal: perbedaan di bawah itu sebanding dengan variasi antar-fold yang lazim. Nilai final ditetapkan di Plan 3 setelah melihat simpangan baku dari uji coba (*pilot*), **bukan** dari hasil utama.
- Tambahkan TOST untuk klaim "tidak berbeda".

---

## Babak 3 — Validitas konstruk replikasi (*strawman*)

**P3.1:** "Kode asli tidak ada. Re-implementasi Anda pasti berbeda. Kalau angka Anda rendah, mungkin implementasi Anda yang jelek, bukan protokol mereka yang salah. Anda menyerang orang-orangan sawah."

**U:** Ini serangan paling penting, dan desainnya bisa menjawab. Inflasi diukur **di dalam implementasi yang sama**: model yang identik dijalankan dengan protokol bocor (L0) dan protokol bersih (L4). Selisih L0 − L4 **hanya disebabkan protokol**, berapa pun tingkat kemiripan implementasi kami dengan aslinya. Kemiripan dengan angka asli hanya menjadi pemeriksaan tambahan: kalau L0 kami mendekati angka yang mereka laporkan, itu bukti re-implementasi wajar.

**P3.2:** "Lilhore punya komponen yang tidak jelas: bias loss, ECA, dilated convolution, cross-layer connection, peran U-Net. Mana yang akan Anda implementasikan?"

**U:** Kami buat **tabel kesetiaan (fidelity table)**. Setiap komponen diberi status: diimplementasikan persis / didekati (dengan asumsinya) / dihilangkan (dengan alasannya). Inti yang wajib ada: MobileNetV3-Large pralatih, input 224, segmentasi lalu crop, Bayesian optimization dengan anggaran tetap. Komponen yang tidak jelas (cross-layer connection) didekati atau dihilangkan, dan itu dinyatakan terang-terangan. Kesimpulan tentang Lilhore dibatasi pada **protokol dan arsitektur intinya**, bukan pada setiap modifikasi mikro.

**Putusan:** ✅ Diterima.
**Perubahan:** tambahkan prinsip "inflasi diukur di dalam implementasi yang sama" dan tabel kesetiaan sebagai luaran wajib Plan 3.

---

## Babak 4 — Confound dalam desain E1

**P4.1:** "Shetty memakai subset 700 citra, sedangkan protokol bersih Anda memakai seluruh 10.015 citra. Anda mengubah dua hal sekaligus, ukuran data dan protokol. Selisihnya tidak bisa diatribusikan."

**U:** Setuju, ini cacat di versi masuk. Revisinya: E1 menjadi **tangga dekomposisi**. Pada tiap anak tangga, hanya **satu** hal yang diubah:

| Anak tangga | Perubahan dari langkah sebelumnya | Mengukur |
|---|---|---|
| L0 | Protokol seperti dilaporkan (augmentasi → split per citra → uji seimbang; Shetty juga melaporkan akurasi latih) | angka "versi paper" |
| L1 | Melaporkan akurasi **uji**, bukan akurasi latih | inflasi akibat pelaporan |
| L2 | Augmentasi/oversampling dipindah **setelah** split (hanya di data latih) | kebocoran salinan augmentasi |
| L3 | Split **per lesi** (+ duplikat hash digabung) | kebocoran lesi saudara |
| L4 | Uji pada **distribusi kelas asli** + metrik BACC | inflasi akibat uji buatan |
| L5 *(tambahan)* | Data latih diperbesar dari subset ke seluruh data (khusus Shetty) | efek ukuran data, **terpisah** dari protokol |

**P4.2:** "Urutan langkah memengaruhi besar kontribusi tiap langkah, karena efeknya bisa berinteraksi."

**U:** Benar. Kami laporkan urutan utama di atas (mengikuti logika pipeline), lalu satu urutan alternatif sebagai **uji sensitivitas**. Bila sumber daya cukup, dekomposisi bisa dibuat simetris dengan rata-rata atas semua urutan (gaya Shapley), tapi itu opsional.

**Putusan:** 🔧 Diterima dengan revisi besar.
**Perubahan:** RQ1 dirumuskan ulang menjadi dekomposisi inflasi L0 → L5, plus satu urutan alternatif sebagai uji sensitivitas.

---

## Babak 5 — Data & unit analisis

**P5.1:** "Satu lesi bisa punya beberapa citra. Kalau satu lesi punya lima citra di data uji, lesi itu berbobot lima kali lipat. Unit analisis Anda citra atau lesi?"

**U:** Analisis utama di tingkat **citra**, karena itu yang dilaporkan semua pembanding dan dipakai ISIC. Analisis sensitivitas di tingkat **lesi**: prediksi dari citra-citra satu lesi dirata-ratakan, lalu dihitung metrik per lesi. Bila kesimpulan berbeda, keduanya dilaporkan.

**P5.2:** "Duplikat dengan `lesion_id` berbeda bisa lolos dari split berkelompok."

**U:** Sudah ada di Plan 1. Duplikat dideteksi dengan *perceptual hash*, lalu digabung ke satu grup sebelum split. Tes otomatis memastikan tidak ada hash yang muncul di dua fold.

**Putusan:** ✅ Diterima.
**Perubahan:** tambahkan analisis sensitivitas tingkat lesi.

---

## Babak 6 — Validitas "uji eksternal"

**P6.1:** "ISIC 2018 Test berasal dari sumber yang sama dengan HAM10000. Itu bukan uji eksternal, hanya hold-out."

**U:** Sebagian benar. Menurut makalah resmi (Codella dkk., 2019), test berisi **1.512 citra = 1.196 internal** (sumber yang sama: Austria dan Australia) **+ 316 eksternal** (Turki, Selandia Baru, Swedia, Argentina, yang tidak ada di data latih). Jadi istilahnya kami perbaiki:
- 1.512 citra = **uji hold-out resmi**
- 316 citra = **partisi eksternal**
- ISIC 2019 (BCN20000) dan PH2 = **uji eksternal sejati** pada tahap publikasi

**P6.2:** "316 citra itu kecil. Berapa melanoma di dalamnya? Interval kepercayaannya pasti lebar."

**U:** Kami hitung kasar. Kalau partisi eksternal berisi ±35 melanoma (**asumsi**, karena distribusi kelas belum diketahui), interval 95% untuk recall 0,70 adalah ±0,15. Artinya partisi eksternal **tidak cukup** untuk membedakan efek kecil. Konsekuensinya:
- H2 (efek CC lebih besar di eksternal) dinyatakan sebagai **analisis eksploratif dengan bootstrap CI**, bukan uji konfirmasi.
- Uji konfirmasi lintas domain dipindah ke tahap publikasi (ISIC 2019, 8.238 citra).

**P6.3:** "Apakah label sumber (internal/eksternal) per citra tersedia publik?"

**U:** **Belum tahu.** Ini risiko nyata. Kalau tidak tersedia, partisi eksternal tidak bisa dianalisis terpisah, dan H2 bergeser sepenuhnya ke tahap publikasi.

**Putusan:** 🔧 Diterima dengan revisi · ❓ P6.3 terbuka.
**Perubahan:**
- Terminologi uji diperbaiki.
- H2 diturunkan statusnya menjadi eksploratif untuk tugas.
- **Tugas Plan 1:** cari metadata sumber per citra test ISIC 2018 dan hitung distribusi kelasnya.

---

## Babak 7 — Metrik

**P7.1:** "Balanced accuracy memberi bobot yang sama ke dermatofibroma dan melanoma. Secara klinis, melewatkan melanoma jauh lebih fatal."

**U:** Setuju. Pembagian perannya:
- BACC tetap jadi **metrik utama**, karena itu metrik resmi ISIC 2018 dan memungkinkan perbandingan dengan 0,885.
- **Sensitivitas melanoma pada spesifisitas tetap** (mis. 80%), dihitung dari skor mel-vs-lainnya, sebagai metrik sekunder yang **klinis**.
- F1 makro, AUROC makro, dan ECE (kalibrasi) sebagai pelengkap.

**P7.2:** "Anda punya banyak metrik. Nanti Anda pilih yang hasilnya bagus."

**U:** Karena itu metrik utama dan sekunder **ditetapkan sekarang** dan dibekukan di Plan 3. Metrik lain diberi label eksploratif.

**Putusan:** 🔧 Diterima dengan revisi.
**Perubahan:** tambahkan metrik klinis "sensitivitas melanoma @ spesifisitas 80%".

---

## Babak 8 — Statistik

**P8.1:** "Anda berencana memakai Wilcoxon pada 5 fold. Tahukah Anda berapa p minimum yang bisa dicapai?"

**U:** Sudah kami hitung: dengan n = 5, **p minimum dua sisi = 0,0625**. Artinya, **tidak mungkin signifikan pada α = 0,05**, sebesar apa pun efeknya. Desain versi masuk cacat. Revisinya:
- **Validasi silang berulang 3×5** (n = 15 pasang skor). Dengan n = 15, p minimum ≈ 0,00006.
- Uji utama: **corrected resampled t-test** (Nadeau & Bengio, 2003), karena fold-fold tidak independen (data latih tumpang tindih). Wilcoxon biasa terlalu optimistis.
- Data uji hold-out: **McNemar** (berpasangan per citra) + **bootstrap 95% CI** untuk selisih BACC.

**P8.2:** "E2 adalah faktorial 2³. Delapan kombinasi × dua model = 16 sel. Kalau semua dibandingkan berpasangan, tingkat temuan palsu Anda meledak."

**U:** E2 tidak dianalisis dengan uji berpasangan antarsel, melainkan dengan **model linear efek campuran**: BACC ~ hair × cc × clahe × model + (1 | fold). Dari model ini dibaca efek utama dan interaksi. Perbandingan berpasangan hanya untuk hipotesis yang sudah ditetapkan, dengan **koreksi Holm**.

**P8.3:** "Berapa efek terkecil yang bisa dideteksi desain Anda?"

**U:** Belum bisa dijawab tanpa simpangan baku antar-fold. **Pilot** di Plan 4.1 akan mengukurnya, lalu dihitung **MDE** (*minimum detectable effect*). Kalau MDE > SESOI, jumlah ulangan ditambah sebelum protokol dibekukan.

**Putusan:** 🔧 Diterima dengan revisi besar · ❓ MDE terbuka (dijawab oleh pilot).
**Perubahan:** 5-fold diganti 3×5 CV berulang, uji Nadeau-Bengio, model efek campuran untuk faktorial, MDE dari pilot sebelum protokol dibekukan.

---

## Babak 9 — Konstruk RQ2 (praproses)

**P9.1:** "Sebagian besar citra HAM10000 tidak berambut. Kalau hair removal dijalankan ke semua citra, efeknya akan terencerkan. Anda bisa menyimpulkan 'tidak berguna', padahal sebenarnya berguna untuk citra berambut."

**U:** Tepat. Rencananya:
1. Profil artefak di Plan 1. Sampel ±300 citra diberi label manual: berambut / tidak, sudut gelap, penggaris.
2. Skor "kerambutan" otomatis (proporsi piksel hasil deteksi black-hat), divalidasi terhadap label manual.
3. Efek hair removal dilaporkan **keseluruhan dan per strata** (berambut vs tidak).

Hipotesisnya pun dibuat spesifik: efek pada strata berambut > efek keseluruhan.

**P9.2:** "*Color constancy* bisa merusak informasi diagnostik. Warna biru-putih (*blue-white veil*) adalah ciri melanoma."

**U:** Ini risiko yang sah dan bisa diuji. Efek CC dilaporkan **per kelas**, terutama recall melanoma. Kalau CC menaikkan BACC tetapi menurunkan recall melanoma, itu temuan penting dan tidak akan disembunyikan.

**P9.3:** "Parameter DullRazor, Shades of Gray (nilai p), dan CLAHE (clip limit) memengaruhi hasil. Kalau Anda men-tuning-nya, Anda curang. Kalau tidak, Anda dituduh memakai parameter jelek."

**U:** Parameter ditetapkan **a priori** dari literatur asal (Shades of Gray p = 6 sebagaimana lazim dipakai; clip limit CLAHE mengikuti Lilhore = 4,0; DullRazor sesuai implementasi rujukan) dan dibekukan di Plan 3. Satu uji sensitivitas parameter kecil dijalankan **hanya di fold validasi dalam**. Nilai-nilai ini masih perlu dikonfirmasi ke sumber primernya di Plan 2.

**Putusan:** 🔧 Diterima dengan revisi.
**Perubahan:** stratifikasi berdasarkan kerambutan, analisis per kelas untuk CC, parameter a priori dengan sumber.

---

## Babak 10 — Keadilan RQ3 (handcrafted vs CNN vs fusi)

**P10.1:** "CNN dapat Bayesian optimization, sedangkan Random Forest pakai default. Itu bukan perbandingan representasi, itu perbandingan usaha tuning."

**U:** Setuju. Semua pipeline mendapat **anggaran tuning yang sama** (jumlah percobaan Optuna yang sama, di fold validasi dalam yang sama). Kepala klasifier juga disamakan untuk perbandingan representasi murni: fitur handcrafted vs embedding CNN beku vs gabungan, semuanya dengan **regresi logistik yang sama**. Setelah itu dilaporkan juga versi terbaik masing-masing (RF untuk handcrafted, fine-tuning untuk CNN).

**P10.2:** "Fusi rawan kebocoran pada normalisasi fitur dan dimensinya timpang (1024 vs ±500)."

**U:** Normalisasi (*scaler*) di-fit per fold hanya pada data latih, dan diimplementasikan sebagai `Pipeline` sklearn supaya tidak mungkin bocor. Ketimpangan dimensi ditangani dengan PCA per blok (di-fit di data latih) atau penskalaan blok, dan pilihannya dibekukan di Plan 3.

**Putusan:** ✅ Diterima.
**Perubahan:** anggaran tuning yang sama, kepala klasifier yang sama untuk perbandingan representasi, dan fusi dalam `Pipeline`.

---

## Babak 11 — Segmentasi (Otsu vs U-Net)

**P11.1:** "U-Net dilatih dengan mask ISIC 2018 Task 1. Citra Task 1 berasal dari arsip ISIC yang sama. Kalau ada lesi uji Anda di sana, segmentasinya bocor."

**U:** Plan 1c mengecek irisan dengan hash. Citra Task 1 yang beririsan dengan **data uji** dikeluarkan dari pelatihan U-Net. Citra yang beririsan dengan data latih fold tertentu tetap aman, selama U-Net dilatih ulang per fold atau irisan dengan semua fold uji dikeluarkan. Pilihan paling sederhana dan aman: **keluarkan semua citra Task 1 yang beririsan dengan HAM10000 + ISIC 2018 Test**.

**P11.2:** "Otsu gagal pada lesi berkontras rendah, dan kegagalan segmentasi akan merusak klasifikasi. Bagaimana Anda tahu penurunan performa berasal dari crop, bukan dari segmentasi yang gagal?"

**U:** Kualitas segmentasi diukur terpisah: Dice/IoU pada Task 1 bagian uji yang bebas irisan, plus **laju kegagalan** (mask kosong atau menutupi seluruh citra). Untuk kasus gagal ada fallback yang ditetapkan di muka: kembali ke citra penuh. Hasil klasifikasi juga dilaporkan terpisah untuk kasus segmentasi berhasil vs gagal.

**Putusan:** ✅ Diterima.
**Perubahan:** aturan pengeluaran irisan Task 1, metrik segmentasi, dan aturan fallback.

---

## Babak 12 — Cakupan & kelayakan

**P12.1:** "RQ1 sampai RQ4, faktorial, 3×5 CV, U-Net, Bayesian optimization, uji eksternal. Ini proyek tesis, bukan tugas kuliah. Anda akan tenggelam."

**U:** Setuju, cakupannya dibagi menjadi tiga tingkat:

| Tingkat | Isi | Untuk |
|---|---|---|
| **MVP (wajib)** | RQ1 dekomposisi untuk **P-S** (Shetty, ringan, CPU) + RQ2 pada **RF-handcrafted dan MobileNetV3** dengan uji hold-out 1.512 | Tugas kuliah |
| **Inti penuh** | + RQ1 untuk P-L (dengan U-Net/Otsu) + RQ3 | Tugas (bila waktu cukup) / dasar publikasi |
| **Publikasi** | + RQ4, ISIC 2019/PH2 lintas domain, lebih banyak seed, kode terbuka | Naskah jurnal |

Perkiraan komputasi MVP: P-S hanya butuh CPU (fitur diekstraksi sekali). RQ2 CNN = 8 kombinasi × 15 run (3×5) ≈ 120 run. Dengan estimasi 10–20 menit per run, totalnya **20–40 jam GPU**. Angka ini perlu diverifikasi di pilot. Kalau terlalu berat, CNN RQ2 bisa diturunkan ke 1×5 CV untuk tugas, tapi konsekuensinya statistik melemah (lihat Babak 8), dan itu dinyatakan sebagai keterbatasan.

**Putusan:** 🔧 Diterima dengan revisi · ❓ tenggat dari dosen belum diketahui.
**Perubahan:** tiga tingkat cakupan dan aturan degradasi yang jujur.

---

## Babak 13 — Etika & sikap ilmiah

**P13.1:** "Anda akan mempublikasikan bahwa dua paper Scientific Reports bermasalah. Itu serius. Bagaimana kalau Anda yang salah membaca?"

**U:** Ada empat pegangan:
1. Setiap kritik merujuk ke **nomor tabel, gambar, dan halaman**, dan hitungan ulangnya ditunjukkan (mis. jumlah sel confusion matrix Lilhore = 11.734; recall tertimbang = akurasi pada Shetty).
2. Nadanya **tentang protokol, bukan tentang orang**.
3. Sebelum naskah publikasi dikirim, **penulis asli dihubungi** untuk klarifikasi. Mungkin ada detail yang tidak tertulis.
4. Kalau ditemukan kesalahan serius, jalur yang tepat dipertimbangkan bersama pembimbing (surat ke editor / *matters arising*, sesuai kebijakan jurnal, yang perlu dicek).

**P13.2:** "Lisensi data?"

**U:** HAM10000 berlisensi CC BY-NC 4.0, sehingga hanya boleh untuk penggunaan nonkomersial dan wajib atribusi. Data tidak diredistribusikan di repositori kami; yang dibagikan hanya ID citra dan file split. Tidak ada subjek manusia baru, jadi tidak perlu persetujuan etik tambahan. Ini tetap perlu dikonfirmasi ke kebijakan departemen.

**Putusan:** ✅ Diterima.
**Perubahan:** kebijakan nada, kontak penulis asli, dan pengaturan redistribusi data.

---

## Babak 14 — Reprodusibilitas kita sendiri

**P14.1:** "Anda mengkritik orang karena tidak membuka kode. Pastikan Anda sendiri bisa direproduksi, termasuk non-determinisme GPU."

**U:**
- Seed ditetapkan (Python/NumPy/PyTorch) dengan `torch.use_deterministic_algorithms(True)` bila memungkinkan, dan sisa non-determinisme cuDNN dicatat.
- Lingkungan dikunci (`uv.lock`).
- Setiap run menyimpan config, commit git, prediksi per citra, dan `metrics.json`. Semua angka di laporan dihasilkan oleh skrip dari prediksi tersimpan, **tidak diketik manual**.
- Protokol dibekukan dengan git tag sebelum hasil utama dilihat.

**Putusan:** ✅ Diterima (sudah sesuai `CLAUDE.md`).

---

## Babak 15 — Serangan terakhir: "Apa satu kalimat kontribusi Anda?"

**P:** "Kalau Anda hanya boleh menulis satu kalimat di abstrak, apa kontribusinya?"

**U:** "Kami menunjukkan berapa poin dari akurasi tinggi yang dilaporkan pada HAM10000 berasal dari pelanggaran protokol, bukan dari metode, dan tahap pengolahan citra mana (penghapusan rambut, *color constancy*, CLAHE, fitur handcrafted) yang benar-benar memberi manfaat ketika dievaluasi dengan benar."

**P:** "Bisa diterima. Pastikan kalimat itu tetap jujur apa pun hasilnya, termasuk kalau jawabannya 'tidak ada yang bermanfaat'."

---

## Rekapitulasi putusan

| Babak | Aspek | Putusan |
|---|---|---|
| 1 | Kebaruan | 🔧 klaim dipertajam (dekomposisi, audit spesifik, ablasi per langkah); CC = uji konfirmasi Barata 2015 |
| 2 | Signifikansi | 🔧 SESOI 2 pp + TOST |
| 3 | Strawman | ✅ inflasi diukur di dalam implementasi yang sama + tabel kesetiaan |
| 4 | Confound E1 | 🔧 tangga dekomposisi L0–L5 + uji sensitivitas urutan |
| 5 | Unit analisis | ✅ + sensitivitas tingkat lesi |
| 6 | Uji eksternal | 🔧 terminologi; H2 eksploratif · ❓ label sumber test |
| 7 | Metrik | 🔧 + sensitivitas melanoma @ spesifisitas 80% |
| 8 | Statistik | 🔧 3×5 CV, Nadeau-Bengio, model efek campuran · ❓ MDE dari pilot |
| 9 | Konstruk RQ2 | 🔧 stratifikasi kerambutan, per kelas, parameter a priori |
| 10 | Keadilan RQ3 | ✅ anggaran tuning sama, kepala klasifier sama |
| 11 | Segmentasi | ✅ aturan irisan, Dice, fallback |
| 12 | Cakupan | 🔧 MVP / inti / publikasi · ❓ tenggat |
| 13 | Etika | ✅ |
| 14 | Reprodusibilitas | ✅ |

**Pertanyaan terbuka yang dibawa ke plan berikutnya:**
- ❓1 (Plan 1) Apakah label sumber internal/eksternal per citra test ISIC 2018 tersedia? Bagaimana distribusi kelas test-nya?
- ❓2 (Plan 1) Jumlah lesi unik HAM10000 (verifikasi angka ±7.470) dan proporsi citra berambut.
- ❓3 (Plan 4.1 → sebelum Plan 3 dibekukan) Simpangan baku antar-fold, MDE, dan waktu per run.
- ❓4 (Dosen) Tenggat, format luaran, dan apakah studi evaluasi ulang/replikasi diterima sebagai tugas.

---

## Lampiran — Fakta & hitungan yang dipakai
| Fakta | Nilai | Sumber / cara |
|---|---|---|
| Komposisi test ISIC 2018 Task 3 | 1.512 = 1.196 internal (Austria, Australia) + 316 eksternal (Turki, Selandia Baru, Swedia, Argentina) | Codella dkk. (2019), arXiv:1902.03368 |
| BACC teratas ISIC 2018 | 0,885 (peringkat 2: 0,882; 3: 0,871) | idem, Tabel 2 |
| Peringkat berubah bila memakai akurasi biasa | Peringkat 4: ACC 0,896 tetapi BACC 0,856 | idem |
| Performa pada sumber di luar data latih lebih rendah | Selisih manusia vs mesin 11,4% (internal) vs 3,6% (eksternal) | Tschandl dkk. (2019), *Lancet Oncol* 20(7):938–947 |
| CC meningkatkan klasifikasi citra multisumber (fitur klasik) | kesimpulan kualitatif | Barata, Celebi & Marques (2015), *IEEE JBHI* 19(3):1146–1152 |
| Wilcoxon p minimum dua sisi | n=5: 0,0625; n=10: 0,00195; n=15: 0,00006 | dihitung (scipy) |
| Lebar CI 95% recall 0,70 | n=171: ±0,069; n=35: ±0,152 | dihitung (aproksimasi normal); jumlah melanoma per partisi masih **asumsi** |
| Koreksi Nadeau-Bengio | varians × (1/n + n_uji/n_latih) | Nadeau & Bengio (2003); paket correctR |
| Inflasi split per citra | BACC +0,05 s/d +0,08; ±34–35% citra uji punya saudara di data latih | LesionBench (GitHub) |
