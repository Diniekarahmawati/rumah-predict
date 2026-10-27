# 🏠 RumahWajar — Prediksi & Cek Kewajaran Harga Rumah Jabodetabek

Proyek Akhir Mata Kuliah **Pembelajaran Mesin** · Tema **Regression**

Aplikasi web untuk:
1. **Mengestimasi harga pasar** rumah di Jabodetabek dari spesifikasinya (luas, kamar, lokasi, sertifikat, dll.).
2. **Mengecek kewajaran** harga yang ditawarkan penjual: *Wajar*, *Di atas pasar*, atau *Di bawah pasar*.
3. **Merekomendasikan area strategis** yang terjangkau di dekat kampus atau kawasan industri.

## 🔗 Aplikasi

| Versi | Model | Link |
|---|---|---|
| Utama | Gradient Boosting | https://prediksi-cari-harga-rumah-jabodetabek.streamlit.app |
| Pembanding | Linear Regression | *(isi link jika sudah di-deploy)* |
| Pembanding | XGBoost | *(isi link jika sudah di-deploy)* |

## 📊 Dataset

[Daftar Harga Rumah Jabodetabek](https://www.kaggle.com/datasets/nafisbarizki/daftar-harga-rumah-jabodetabek) (Kaggle), hasil scraping iklan rumah123.com.
- 3.553 iklan, 27 kolom → setelah dibersihkan **3.153 rumah** di 9 kota/kabupaten.
- Target: harga (Rupiah), dimodelkan dalam skala **log** karena distribusinya sangat miring (skewness 24,7 → 0,66).
- Split 80:20 (2.522 latih, 631 uji), stratified per kota.

## 🤖 Hasil Model (data uji, 631 rumah)

| Model | MAPE | MAE | R² (log) |
|---|---|---|---|
| Baseline (tebak median) | 94,24% | – | – |
| Linear Regression | 29,53% | Rp1,21 M | 0,892 |
| Random Forest | 24,95% | Rp1,09 M | 0,907 |
| Gradient Boosting | 22,95% | Rp0,90 M | 0,919 |
| XGBoost | 22,18% | Rp0,91 M | 0,923 |

*MAPE = rata-rata persentase kesalahan prediksi (makin kecil makin baik). Hyperparameter dituning dengan Grid Search + 5-Fold Cross Validation.*

## 📍 Pengembangan: Analisis Lokasi Strategis

Model dipakai untuk memprediksi harga **rumah standar 70/90** (LB 70 m², LT 90 m², 2 KT, 1 KM) di setiap area, sehingga harga antar-area bisa dibandingkan secara adil. Lalu dihitung jaraknya ke **16 kampus** dan **10 kawasan industri**.

Temuan utama:
- Makin dekat kampus, harga cenderung makin mahal (korelasi Spearman −0,49).
- Kedekatan dengan kawasan industri hampir tidak berpengaruh pada harga (korelasi +0,14), sehingga area di sekitarnya relatif terjangkau.

## 📁 Struktur Repository

```
├── final_model.py                    # aplikasi utama (Gradient Boosting)
├── app_linear_regression.py          # aplikasi versi Linear Regression
├── app_xgboost.py                    # aplikasi versi XGBoost
├── requirements.txt                  # library untuk Streamlit
├── model_terbaik.joblib              # model Gradient Boosting (pipeline preprocessing + model)
├── model_linear_regression.joblib    # model Linear Regression
├── model_xgboost.joblib              # model XGBoost
├── info_model*.json                  # metrik & daftar fitur tiap model
├── perbandingan_model.json           # metrik semua model untuk tab Perbandingan
├── lokasi_district.csv               # area/kecamatan → koordinat
├── harga_per_area.csv                # harga rumah standar per area
├── titik_acuan.csv                   # koordinat kampus & kawasan industri
├── ringkasan_titik_acuan.csv         # harga di sekitar tiap kampus/kawasan industri
├── .streamlit/config.toml            # tema warna aplikasi
└── notebooks/
    └── Proyek_ML_Harga_Rumah_Lengkap.ipynb   # seluruh tahapan: EDA → preprocessing → modeling → lokasi
```

## ▶️ Menjalankan di Komputer Sendiri

```bash
pip install -r requirements.txt
streamlit run final_model.py
```

## ⚠️ Keterbatasan

- Harga di data adalah **harga iklan**, bukan harga transaksi.
- Lokasi memakai titik tengah kecamatan/area, bukan alamat persis.
- Estimasi ini **bukan penilaian (appraisal) resmi**.

## 👥 Tim

| Nama | NIM | Peran |
|---|---|---|
| Dini Eka Rahmawati | J0404241031 | Analis Data |
| Siti Lativa Irmiyati | J0404231003 | Data Engineer & Modeler |
| Lukiman Hakim | J0404241045 | Evaluator & Developer |
