# Panduan Penggunaan Aplikasi

Dokumen ini menjelaskan cara menjalankan dan menguji aplikasi web Streamlit untuk prediksi risiko diabetes berbasis perbandingan tiga pendekatan:
- FIS Manual
- FIS + Genetic Algorithm (GA)
- ANFIS (Neuro-Fuzzy)

## Persyaratan Sistem

### Versi Dasar
- Python 3.10 atau lebih baru
- pip (package manager Python)

### Library Utama
- streamlit
- numpy
- pandas
- matplotlib
- seaborn
- scikit-fuzzy
- scikit-learn
- torch

Semua dependensi beserta rentang versinya sudah tersedia pada file requirements.txt.

## Langkah Instalasi dan Akses Aplikasi

### 1. Buka folder proyek
Pastikan terminal berada di direktori proyek:

```powershell
cd C:\Coding\Softcom
```

### 2. Install dependensi

```powershell
pip install -r requirements.txt
```

### 3. Jalankan aplikasi

```powershell
streamlit run app.py
```

### 4. Akses di browser
Setelah perintah dijalankan, Streamlit akan menampilkan URL lokal (biasanya http://localhost:8501). Buka URL tersebut di browser.

## Fungsi Utama Antarmuka (UI)

Aplikasi memiliki tiga tab utama:

### 1. Simulator Prediksi
Tab ini digunakan untuk input parameter pasien dan menampilkan hasil prediksi tiga model.

Langkah penggunaan:
1. Isi seluruh parameter klinis pada form.
2. Klik tombol Prediksi Risiko.
3. Baca probabilitas risiko dan kategori risiko untuk masing-masing model.
4. Gunakan rata-rata skor gabungan sebagai ringkasan cepat.

### 2. EDA
Tab ini menampilkan:
- Ringkasan performa model (berdasarkan parameter model yang tersimpan)
- Preview dataset
- Visualisasi EDA (distribusi kelas outcome dan korelasi antar fitur)

Tujuan tab ini adalah memberi konteks data dan performa sebelum melakukan simulasi prediksi.

### 3. Analisis Kurva MF
Tab ini berfokus pada perbandingan kurva Membership Function (MF) antar model:
- FIS Manual
- FIS + GA
- ANFIS (Gaussian MF)

Tab ini digunakan untuk memahami perbedaan karakter kurva dan implikasinya terhadap sensitivitas prediksi.

## Struktur File Penting

- app.py: Entry point aplikasi Streamlit
- diabetes_app/: Modul aplikasi (tema, loader, inferensi, visualisasi, halaman)
- diabetes.csv: Dataset mentah untuk EDA
- model/mf_params.json: Parameter MF untuk FIS Manual, FIS + GA, dan parameter ANFIS
- model/anfis_model.pkl: Bundle model ANFIS (state_dict, scaler, metadata)

## Catatan Pengujian

Untuk pengujian cepat:
1. Jalankan aplikasi.
2. Ubah nilai parameter di tab Simulator Prediksi.
3. Pastikan ketiga model menghasilkan skor.
4. Buka tab Analisis Kurva MF dan verifikasi kurva untuk Glucose, BMI, dan Age tampil normal.
