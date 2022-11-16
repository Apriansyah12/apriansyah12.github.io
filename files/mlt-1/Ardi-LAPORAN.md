# Laporan Proyek Machine Learning - Muhamad Ardi Apriansyah

## Project Overview
Perusahaan asuransi kesehatan berencana membuat fitur untuk mempermudah pengguna dalam melihat skema pembiayaan setelah menjadi pengguna asuransi. Sehingga pengguna lebih siap untuk membuat *financial planning* kedepannya. Dan pembiayaan dapat terlaksana dengan lancar.
Kaushik, dkk (2022) melakukan penelitian untuk memprediksi biaya asuransi kesehatan (individual) menggunakan algoritma ANN (regresi) mendapatkan hasil akurasi sebesar 92.72% [[1]](https://www.mdpi.com/1660-4601/19/13/7898). *Artificial Neural Network* (ANN) merupakam salah satu metode dalam Machine Learning. Sejalan dengan penelitian tersebut, solusi yang ditawarkan yaitu menggunakan pendekatan *Machine Learning* dengan metode KNN, Random Forest dan AdaBoosting untuk melakukan prediksi biaya asuransi kesehatan yang ditanggung pengguna baru.

## Business Understanding

### Problem Statements
- Perusahaan asuransi kesahatan membutuhkan model terbaik untuk melakukan prediksi biaya asuransi yang akan ditanggung pengguna baru.

### Goals
- Membangun model terbaik untuk melakukan prediksi biaya asuransi yang akan ditanggung pengguna baru.

### Solution statements.
- Menawarkan solusi sistem prediksi dengan metode regresi. Untuk mendapatkan solusi terbaik, akan digunakan tiga model yang berbeda (KNN, Random Forest, AdaBoosting) dengan *Hyperparameter tuning*. Selain itu, untuk mengukur kinerja model digunakan metrik MSE (*Mean Squared Error*). Dimana model terbaik nantinya harus memperoleh nilai MSE terendah dari data uji.

## Data Understanding

Tabel 1. Informasi Dataset

| | Keterangan |
|---|---|
| Sumber | [Kaggle - Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) |
| Jumlah Data | 1338 |
| *Usability* | 8.82 |
| Lisensi | [Lisensi](http://opendatacommons.org/licenses/dbcl/1.0/) |
| *Rating* | *gold* |
| Jenis dan Ukuran Berkas | csv (16 kB) |


### Variabel-variabel pada Dataset
Berdasarkan informasi dari [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance), variabel-variabel pada Diamond dataset adalah sebagai berikut:

- age: merepresentasikan usia pengguna
- sex: merepresentasikan jenis kelamin pengguna
- bmi: merepresentasikan berat badan pengguna
- children: merepresentasikan jumlah anak yang dimiliki
- smoker: merepresentasikan status perokok atau bukan perokok
- region: merepresentasikan wilayah tempat tinggal pengguna
- charges: biaya medis pengguna yang ditagih oleh asuransi kesehatan

### Menangani Missing Value
Untuk mendeteksi *missing value* digunakan fungsi isnull().sum() dan diperoleh:

Tabel 2. Hasil Deteksi *Missing Value*

| Kolom | Jumlah *Missing Value* |
|---|:---:|
| age | 0 |
| sex | 0 |
| bmi | 0 |
| children | 0 |
| smoker | 0 |
| region | 0 |
| charges | 0 |

Dari Tabel 2 di atas, terlihat bahwa setiap fitur tidak memiliki *missing value*.

### Menangani Outlier
Pada kasus ini, untuk mendeteksi *outliers* digunakan teknis visualisasi data (boxplot). Kemudian untuk menangani *outliers* digunakan metode IQR.

Seltman dalam “Experimental Design and Analysis” [2] menyatakan bahwa outliers yang diidentifikasi oleh boxplot (disebut juga “boxplot *outliers*”) didefinisikan sebagai data yang nilainya 1.5 IQR di atas Q3 atau 1.5 IQR di bawah Q1.

Berikut persamaannya:
```
Batas bawah = Q1 - 1.5 * IQR
Batas atas = Q3 + 1.5 * IQR
```
Tabel 3. Visualisasi Boxplot Sebelum dan Sesudah Dikenakan Metode IQR.

| Cek Outlier Pada Fitur | Setelah Dikenakan Metode IQR |
|:---:|:---:|
| Fitur age (Before) ![](https://apriansyah12.github.io/images/mlt-1/age-before.png) | Fitur age (After) ![](https://apriansyah12.github.io/images/mlt-1/age-after.png) |
| Fitur bmi (Before) ![](https://apriansyah12.github.io/images/mlt-1/bmi-before.png) | Fitur bmi (After) ![](https://apriansyah12.github.io/images/mlt-1/bmi-after.png) |
| Fitur children (Before) ![](https://apriansyah12.github.io/images/mlt-1/children-before.png) | Fitur children (After) ![](https://apriansyah12.github.io/images/mlt-1/children-after.png) |
| Fitur charges (Before) ![](https://apriansyah12.github.io/images/mlt-1/charges-before.png) | Fitur charges (After) ![](https://apriansyah12.github.io/images/mlt-1/charges-after.png) |

Dari hasil deteksi ulang outlier dengan boxplot di Tabel 3 di atas, didapat bahwa outlier sudah berkurang pada tiap fitur setelah dibersihkan seperti pada Tabel 4. berikut.

Tabel 4. Perbandingan Jumlah Data Sebelum dan Setelah Dibersihkan dari Outlier

| Jumlah Data Sebelum Dibersihkan | Jumlah Data Setelah Dibersihkan |
|:---:|:---:|
| 1338 | 1193 |

### Univariate Analysis

#### Fitur Kategorik

##### Fitur sex
![](https://apriansyah12.github.io/images/mlt-1/sex-kategori.png)

Gambar 1. Kategori pada Fitur sex

Dari Gambar 1. terdapat 2 kategori pada fitur sex, secara berurutan dari jumlahnya paling banyak yaitu: female (611) dan male (582).

##### Fitur smoker
![](apriansyah12.github.io/images/mlt-1/smoker-kategori.png)

Gambar 2. Kategori pada Fitur smoker

Dari Gambar 2. terdapat 2 kategori pada fitur smoker, secara berurutan dari jumlahnya paling banyak yaitu: no (1055) dan yes (138). Dari data presentase dapat kita simpulkan bahwa lebih dari 80% merupakan bukan perokok.

##### Fitur region
![](https://apriansyah12.github.io/images/mlt-1/region-kategori.png)

Gambar 3. Kategori pada Fitur region

Dari Gambar 3. terdapat 4 kategori pada fitur region, secara berurutan dari jumlahnya paling banyak yaitu: northwest (305), southeast (302), northeast (295), southwest (291).

#### Fitur Numerik
Selanjutnya, untuk fitur numerik, kita akan melihat histogram masing-masing fiturnya menggunakan code berikut.

![](apriansyah12.github.io/images/mlt-1/histogram-numerik.png)

Gambar 4. Histogram pada Setiap Fitur Numerik

Berdasarkan Gambar 4. di atas, diperoleh beberapa informasi, antara lain:

- Pada histogram age terdapat peningkatan jumlah pengguna dijenjang umur kurang dari 20 tahun
- Pada histogram charges miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model.

### Multivariate Analysis

#### Fitur Kategorik

##### Fitur sex
![](https://apriansyah12.github.io/images/mlt-1/charges-sex.png)

Gambar 5. Hubungan Fitur charges dengan Fitur sex

Dari Gambar 5. pada fitur sex, rata-rata charges cenderung mirip, rentangnya berada antara 9500 hingga 10000. Sehingga fitur sex memiliki pengaruh atau dampak yang kecil.

##### Fitur smoker
![](https://apriansyah12.github.io/images/mlt-1/charges-smoker.png)

Gambar 6. Hubungan Fitur charges dengan Fitur smoker

Dari Gambar 6. pada fitur smoker, rata-rata charges pada perokok lebih dari 20000. Hal ini jauh lebih tinggi dari rata-rata charges pada non-perokok yang berada di sekitar 7000. Sehingga fitur smoker memiliki pengaruh atau dampak yang cukup besar.

##### Fitur Region
![](https://apriansyah12.github.io/images/mlt-1/charges-region.png)

Gambar 7. Hubungan Fitur charges dengan Fitur region

Dari Gambar 7. pada fitur region, rata-rata charges cenderung mirip, rentangnya berada antara 9000 hingga 10000. Sehingga fitur region memiliki pengaruh atau dampak yang kecil.

#### Fitur Numerik
Untuk mengamati hubungan antara fitur numerik, akan digunakan fungsi pairplot(), dengan output sebagai berikut.

![](https://apriansyah12.github.io/images/mlt-1/pairplot-numerik.png)

Gambar 8. Visualisasi Hubungan antar Fitur Numerik

Pada pola sebaran data grafik pairplot di atas, terlihat fitur age memiliki korelasi cukup kuat (positif) dengan fitur charges (target). Untuk mengevaluasi skor korelasinya, akan digunakan fungsi corr() sebagai berikut.

![](https://apriansyah12.github.io/images/mlt-1/corr-numerik.png)

Gambar 9. Korelasi antar Fitur Numerik

Koefisien korelasi berkisar antara -1 dan +1. Semakin dekat nilainya ke 1 atau -1, maka korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0 maka korelasinya semakin lemah.

Dari grafik korelasi di atas, fitur age memiliki korelasi yang cukup kuat (0.44) dengan fitur target charges.

## Data Preparation

### Encoding Fitur Kategori

Untuk melakukan proses encoding fitur kategori, salah satu teknik yang umum dilakukan adalah teknik *one-hot-encoding*. Library scikit-learn menyediakan fungsi ini untuk mendapatkan fitur baru yang sesuai sehingga dapat mewakili variabel kategori. Kita memiliki tiga variabel kategori dalam dataset kita, yaitu 'sex', 'smoker', dan 'region'. Mari kita lakukan proses *encoding* ini dengan fitur *get_dummies*.

Tabel 5. Hasil *One-Hot-Encoding*

|index|age|bmi|children|charges|sex\_female|sex\_male|smoker\_no|smoker\_yes|region\_northeast|region\_northwest|region\_southeast|region\_southwest|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|19|27\.9|0|16884\.924|1|0|0|1|0|0|0|1|
|1|18|33\.77|1|1725\.5523|0|1|1|0|0|0|1|0|
|2|28|33\.0|3|4449\.462|0|1|1|0|0|0|1|0|
|3|33|22\.705|0|21984\.47061|0|1|1|0|0|1|0|0|
|4|32|28\.88|0|3866\.8552|0|1|1|0|0|1|0|0|

### Reduksi Dimensi dengan PCA

PCA umumnya digunakan ketika variabel dalam data yang memiliki korelasi yang tinggi. Korelasi tinggi ini menunjukkan data yang berulang atau redundant. Sebelumnya perlu cek kembali korelasi antar fitur (selain fitur target) dengan menggunakan pairplot.

![](https://apriansyah12.github.io/images/mlt-1/pairplot-all.png)

Gambar 10. Visualisasi Hubungan antar Fitur Selain Fitur Target (charges)

Selanjutnya kita akan mereduksi region_northeast, region_northwest,  region_southeast, region_southwest karena keempatnya berkorelasi cukup kuat yang dapat dilihat pada visualisasi pairplot di atas.

Untuk implementasinya menggunakan fungsi PCA() dari sklearn dengan mengatur nilai parameter n_components sebanyak fitur yang akan dikenakan PCA.

Tabel 6. Proporsi *Principal Component* dari Hasil PCA

| PC Pertama | PC Kedua | PC Ketiga | PC Keempat |
|:---:|:---:|:---:|:---:|
| 0.339 | 0.333 | 0.327 | 0 |

Arti dari Tabel 6. di atas adalah, 33.9% informasi pada keempat fitur ( region_northeast, region_northwest, region_southeast, region_southwest ) terdapat pada PC (Principal Component) pertama. Sedangkan sisanya sebesar 33.3% terdapat pada PC kedua dan 32.7% pada PC ketiga.

Berdasarkan hasil tersebut, akan dilakukan reduksi fitur dan hanya mempertahankan PC (komponen) pertama, kedua dan ketiga. Ketiga PC ini akan menjadi fitur yang menggantikan empat fitur lainnya ( region_northeast, region_northwest, region_southeast, region_southwest ). Fitur-fitur ini akan diberi nama PCA_region_1, PCA_region_2, PCA_region_3.

Tabel 7. Tampilan 5 Sampel dari Dataset Setelah Dilakukan Reduksi Fitur

|index|age|bmi|children|charges|sex\_female|sex\_male|smoker\_no|smoker\_yes|PCA\_region\_1|PCA\_region\_2|PCA\_region\_3|
|---|---|---|---|---|---|---|---|---|---|---|---|
|33|63|28\.31|0|13770\.0979|0|1|1|0|-0\.6015303722290504|-0\.6019243502683033|-0\.1401708311781894|
|892|54|24\.035|0|10422\.91665|0|1|1|0|-0\.6015303722290504|-0\.6019243502683033|-0\.1401708311781894|
|280|40|28\.12|1|22331\.5668|1|0|0|1|-0\.11751229861251355|0\.6616950366574698|-0\.551257865026094|
|1137|26|22\.23|0|3176\.2877|1|0|1|0|-0\.6015303722290504|-0\.6019243502683033|-0\.1401708311781894|
|618|19|33\.11|0|34439\.8559|1|0|0|1|0\.7860771547541603|-0\.33101318266693885|-0\.1062224832301332|

### Train Test Split

Pada tahap ini akan dibagi dataset menjadi data latih (train) dan data uji (test). Pada kasus ini akan menggunakan proporsi pembagian sebesar 80:20 dengan fungsi train_test_split dari sklearn.

Tabel 7. Jumlah Data Latih dan Uji

| Jumlah Data Latih | Jumlah Data Uji | Jumlah Total Data |
|:---:|:---:|:---:|
| 851 | 213 | 1064 |

Catatan: angka 1064 didapat setelah dilakukan pembersihan nilai NaN sebelum dilakukan split dataset.

### Standarisasi

Proses standarisasi bertujuan untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn.

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandarScaler menghasilkan distribusi deviasi sama dengan 1 dan mean sama dengan 0.

Tabel 8. Hasil Proses Standarisasi pada Setiap Fitur

|index|age|bmi|children|sex\_female|sex\_male|smoker\_no|smoker\_yes|PCA\_region\_1|PCA\_region\_2|PCA\_region\_3|
|---|---|---|---|---|---|---|---|---|---|---|
|51|-1\.329345795648075|0\.5960927016343016|0\.797472291511087|0\.9687603331802496|-0\.9687603331802496|0\.3565842367250616|-0\.3565842367250615|1\.5434580524671695|-0\.6501033802305207|-0\.2043887927498102|
|869|-1\.0454451062456|-0\.979169310419975|1\.6415629881229463|0\.9687603331802496|-0\.9687603331802496|0\.3565842367250616|-0\.3565842367250615|-1\.187842586183341|-1\.1910994908848527|-0\.2735660498422037|
|613|-0\.4066685550900316|-1\.8740126827552017|1\.6415629881229463|0\.9687603331802496|-0\.9687603331802496|0\.3565842367250616|-0\.3565842367250615|-0\.23512443794629118|1\.3322854981022587|-1\.1112466778173342|
|630|0\.9418597195717242|1\.013123480816945|-0\.0466184051007725|-1\.0322470540440039|1\.0322470540440039|0\.3565842367250616|-0\.3565842367250615|-0\.23512443794629118|1\.3322854981022587|-1\.1112466778173342|
|1126|1\.0838100642729616|-0\.0336744264431322|-0\.8907091017126318|-1\.0322470540440039|1\.0322470540440039|0\.3565842367250616|-0\.3565842367250615|-1\.187842586183341|-1\.1910994908848527|-0\.2735660498422037|

## Modeling
Pada tahap ini, kita akan menggunakan tiga algoritma untuk kasus regresi ini. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menetukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan kita gunakan, antara lain:

1. K-Nearest Neighbor

    Kelebihan algoritma KNN adalah mudah dipahami dan digunakan sedangkan kekurangannya kika dihadapkan pada jumlah fitur atau dimensi yang besar rawan terjadi bias.

2. Random Forest
    
    Kelebihan algoritma Random Forest adalah menggunakan teknik Bagging yang berusaha melawan *overfitting* dengan berjalan secara paralel. Sedangkan kekurangannya ada pada kompleksitas algoritma Random Forest yang membutuhkan waktu relatif lebih lama dan daya komputasi yang lebih tinggi dibanding algoritma seperti Decision Tree.

3. Boosting Algorithm

    Kelebihan algoritma Boosting adalah menggunakan teknik Boosting yang berusaha menurunkan bias dengan berjalan secara sekuensial (memperbaiki model di tiap tahapnya). Sedangkan kekurangannya hampir sama dengan algoritma Random Forest dari segi kompleksitas komputasi yang menjadikan waktu pelatihan relatif lebih lama, selain itu *noisy* dan *outliers* sangat berpengaruh dalam algoritma ini.

Langkah pertama membuat DataFrame baru df_models untuk menampung nilai metrik pada setiap model / algoritma. Hal ini berguna untuk melakukan analisa perbandingan antar model. Metrik yang digunakan untuk mengevaluasi model adalah (MSE - Mean Squared Error).

### Model KNN
KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih k tetangga terdekat. Pemilihan nilai k sangat penting dan berpengaruh terhadap performa model. Jika memilih k yang terlalu rendah, maka akan menghasilkan model yang *overfitting* dan hasil prediksinya memiliki varians tinggi. Sedangkan jika memilih k yang terlalu tinggi, maka model yang dihasilkan akan *underfitting* dan prediksinya memiliki bias yang tinggi [[2]](https://www.oreilly.com/library/view/machine-learning-with/9781617296574/).

Oleh karena itu, perlu mencoba beberapa nilai k yang berbeda (1 sampai 20) kemudian membandingan mana yang menghasilkan nilai metrik model (pada kasus ini memakai *mean squared error*) terbaik. Selain itu, akan digunakan metrik ukuran jarak secara *default* (Minkowski Distance) pada KNeighborsRegressor dari *library* sklearn.

Tabel 9. Perbandingan Nilai K terhadap Nilai MSE

| K | MSE |
|:---:|---|
| 1 | 44207836.988966085 |
| 2 | 31538753.857456606 |
| 3 | 28674920.86315255 |
| 4 | 27629265.834566347 |
| 5 | 25390953.014612235 |
| 6 | 25518720.88723256 |
| 7 | 24308852.57983258 |
| 8 | 23648019.45785829 |
| 9 | 23717372.86834201 |
| 10 | 23786877.344998296 |
| 11 | 23827083.011642676 |
| 12 | 23636700.67855662 |
| 13 | 23457928.835207216 |
| 14 | 23285642.54610008 |
| 15 | 23340812.39508889 |
| 16 | 23272963.082737543 |
| 17 | 22840271.478609767 |
| 18 | 22987447.387531828 |
| 19 | 22663737.78903354|
| 20 | 22851240.315473415 |

Jika divisualisasikan dengan fungsi `plot()` diperoleh:

![](https://apriansyah12.github.io/images/mlt-1/tuning-knn.png)

Gambar 11. Visualisai Nilai K terhadap MSE

Dari hasil output diatas, nilai MSE terbaik dicapai ketika k = 17 yaitu sebesar 22840271.4786. Oleh karena itu kita akan menggunakan k = 17 dan menyimpan nilai MSE nya (terhadap data latih, untuk data uji akan dilakukan pada proses evaluasi) kedalam df_models yang telah kita siapkan sebelumnya.

### Model Random Forest

Random forest merupakan algoritma *supervised learning* yang termasuk ke dalam kategori *ensemble* (group) learning. Pada model *ensemble*, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model *ensemble* ini digabungkan untuk membuat prediksi akhir. Jenis metode *ensemble* yang digunakan pada Random Forest adalah teknik *Bagging*. Metode ini bekerja dengan membuat subset dari data train yang independen. Beberapa model awal (base model / weak model) dibuat untuk dijalankan secara simultan / paralel dan independen satu sama lain dengan subset data train yang independen. Hasil prediksi setiap model kemudian dikombinasikan untuk menentukan hasil prediksi final.

Kita akan menggunakan `RandomForestRegressor` dari *library* scikit-learn dengan base_estimator defaultnya yaitu DecisionTreeRegressor dan parameter-parameter (hyperparameter) yang digunakan antara lain:

- n_estimator: jumlah trees (pohon) di forest.
- max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.
- random_state: digunakan untuk mengontrol random number generator yang digunakan.
- n_jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.

Untuk menentukan nilai *hyperparameter* (n_estimator & max_depth) di atas, akan dilakukan *tuning* dengan GridSearchCV.

Tabel 10. Hasil *Hyperparameter Tuning* model *GridSearchCV* dengan Random Forest

| | Daftar Nilai | Nilai Terbaik |
|---|---|---|
| n_estimators | 10, 20, 30, 40, 50, 60, 70, 80, 90 | 90 |
| learning_rate | 4, 8, 16, 32 | 4 |
| MSE data latih | | 17568673.086 |
| MSE data uji | |19174816.288 |

Dari hasil output di atas diperoleh nilai MSE terbaik dalam jangkauan parameter params_rf yaitu 17568673.086 (dengan data train) dan 19174816.288 (dengan data test) dengan n_estimators: 90 dan max_depth: 4. Selanjutnya kita akan menggunakan pengaturan parameter tersebut dan menyimpan nilai MSE nya kedalam df_models yang telah kita siapkan sebelumnya.

### Model AdaBoosting

Jika sebelumnya kita menggunakan algoritma *bagging* (Random Forest). Selanjutnya kita akan menggunakan metode lain dalam model *ensemble* yaitu teknik *Boosting*. Algoritma *Boosting* bekerja dengan membangun model dari data train. Kemudian membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Teknik ini bekerja secara sekuensial.

Pada kasus ini kita akan menggunakan metode *Adaptive Boosting*. Untuk implementasinya kita menggunakan AdaBoostRegressor dari library sklearn dengan base_estimator defaultnya yaitu DecisionTreeRegressor hampir sama dengan RandomForestRegressor bedanya menggunakan metode teknik *Boosting*.

Parameter-parameter (hyperparameter) yang digunakan pada algoritma ini antara lain:

- n_estimator: jumlah *estimator* dan ketika mencapai nilai jumlah tersebut algoritma Boosting akan dihentikan.
- learning_rate: bobot yang diterapkan pada setiap *regressor* di masing-masing iterasi Boosting.
- random_state: digunakan untuk mengontrol *random number* generator yang digunakan.

Untuk menentukan nilai *hyperparameter* (n_estimator & learning_rate) di atas, kita akan melakukan *tuning* dengan GridSearchCV.

Tabel 11. Hasil *Hyperparameter Tuning* model *GridSearchCV* dengan AdaBoosting

| | Daftar Nilai | Nilai Terbaik |
|---|---|---|
| n_estimators | 10, 20, 30, 40, 50, 60, 70, 80, 90 | 50 |
| learning_rate | 0.001, 0.01, 0.1, 0.2 | 0.01 |
| MSE data latih | | 18874829.086 |
| MSE data uji | | 19635802.344 |

Dari hasil output di atas diperoleh nilai MSE terbaik dalam jangkauan parameter params_ab yaitu 18874829.086 (dengan data train) dan 19635802.344 (dengan data test) dengan n_estimators: 50 dan learning_rate: 0.01. Selanjutnya kita akan menggunakan pengaturan parameter tersebut dan menyimpan nilai MSE nya kedalam df_models yang telah kita siapkan sebelumnya.

## Evaluation
Dari proses sebelumnya, kita telah membuat tiga model yang berbeda dan juga telah melatihnya. Selanjutnya kita perlu mengevaluasi model-model tersebut menggunakan data uji dan metrik yang digunakan dalam kasus ini yaitu mean_squared_error. Hasil evaluasi kemudian kita simpan ke dalam df_models.

$$\text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2.$$

Dengan:
- $n_{\text{sample}}$ adalah banyaknya data
- $\hat{y}_i$ adalah hasil prediksi sedangkan $y_i$ adalah nilai yang akan diprediksi (nilai yang sebenarnya).

Berdasarkan DataFrame `df_models` diperoleh:

Tabel 12. Nilai MSE pada Setiap Model dengan Data Latih dan Data Uji

|index|KNN|RandomForest|Boosting|
|---|---|---|---|
|Train MSE|20893830\.721873555|17583824\.71471183|21386903\.89729386|
|Test MSE|22840271\.478609767|18797803\.324574962|21631276\.058484707|

Untuk memudahkan, dilakukan *plot* hasil evaluasi model dengan *bar chart* sebagai berikut:

![](https://apriansyah12.github.io/images/mlt-1/evaluasi-model.png)

Gambar 12. *Bar Chart* Hasil Evaluasi Model dengan Data Latih dan Uji

Dari Tabel 12. dan Gambar 12. di atas, terlihat bahwa, model Random Forest memberikan nilai MSE terendah (18797803.324) pada data uji. Sedangkan model KNN memiliki MSE tertinggi (22840271.478) pada data uji. Sebelum memutuskan model terbaik untuk melakukan prediksi biaya asuransi yang ditanggung pengguna. Perlu dilakukan uji prediksi menggunakan beberapa sampel acak (5) pada data uji dengan hasil sebagai berikut.

Tabel 13. Hasil Prediksi dari 5 Sampel Acak

|index\_sample|y\_true|prediksi\_KNN|prediksi\_RF|prediksi\_Boosting|
|---|---|---|---|---|
|237|4463\.2051|7148\.022531176472|6010\.452935658891|8162\.155537480002|
|403|10269\.46|11223\.06748882353|11502\.196700265602|13514\.180115530953|
|132|11163\.568|12899\.519489411765|12383\.58872438205|13514\.180115530953|
|518|5240\.765|6820\.887846470589|6886\.607595197752|8176\.287071584702|
|664|27037\.9141|23587\.395952941177|25471\.551143508907|23895\.02001131579|
|768|14319\.031|12162\.10210117647|14987\.491836788236|15104\.696844857142|
|750|19539\.243|22935\.98669411765|19594\.504246816166|20508\.369218333333|
|336|12142\.5786|11380\.24352235294|15234\.154099635021|15104\.696844857142|
|482|1622\.1885|3905\.273577|3304\.906370947735|5585\.758352763816|
|778|5934\.3798|7723\.563858823529|6933\.147957891633|8182\.845100801401|

Dari Tabel 13, terlihat bahwa prediksi dengan Random Forest memberikan hasil yang paling mendekati.

## Conclusion
Berdasarkan hasil evaluasi model di atas, dapat disimpulkan bahwa model terbaik untuk melakukan prediksi biaya asuransi kesehatan adalah model Random Forest. Hal ini dilihat dari nilai MSE pada data uji yang menunjukan bahwa algoritma Boosting mempunyai MSE terendah sebesar 17583824.714 (pada data latih) dan 18797803.324 (pada data uji).

## Daftar Referensi
[1] Kaushik, K., Bhardwaj, A., Dwivedi, A.D. and Singh, R., 2022. Machine Learning-Based Regression Framework to Predict Health Insurance Premiums. International Journal of Environmental Research and Public Health, [online] 19(13). https://doi.org/10.3390/ijerph19137898. Tersedia: [tautan](https://www.mdpi.com/1660-4601/19/13/7898#cite).  
[2] Rhys, Hefin. "Machine Learning with R, the Tidyverse, and MLR". Manning Publications. 2020. Page 286. Tersedia: [O'Reilly Media](https://learning.oreilly.com/library/view/machine-learning-with/9781617296574/).