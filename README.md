# 🪙 Bitcoin Multi-Horizon Forecasting using Deep Learning

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

Proyek ini merupakan submission akhir untuk mata kuliah **Deep Learning untuk Time Series (DLTM)**. Fokus proyek ini adalah melakukan peramalan harga Bitcoin secara *multi-horizon* (24 jam ke depan) menggunakan pendekatan Deep Learning dengan arsitektur **LSTM** dan **Seq2Seq (Sequence-to-Sequence)**.

---

## 🚀 Fitur Utama

- **Multi-Horizon Forecasting**: Memprediksi 24 langkah (*step*) ke depan secara simultan menggunakan data historis harga Bitcoin.
- **Weighted Horizon Loss**: Menggunakan fungsi loss kustom di mana bobot *error* meningkat pada setiap *step* proyeksi masa depan (dari bobot `1.0` di Step 1 hingga `3.3` di Step 24).
- **Custom Training Loop dengan Akselerasi `@tf.function`**: Proses pelatihan model dipercepat hingga 10-20x menggunakan kompilasi graf statis TensorFlow.
- **Kombinasi 3 Model Deep Learning**:
  1. **Baseline LSTM**
  2. **Seq2Seq LSTM Subclass (Autoregressive)**
  3. **Seq2Seq Functional**
- **Custom Callbacks**: Implementasi kustom untuk *Early Stopping* dan *Learning Rate Reducer* yang kompatibel dengan Keras 3.

---

## 📊 Dataset

Dataset yang digunakan adalah [bitcoin_dataset.csv](file:///c:/Users/majan/.gemini/antigravity-ide/scratch/DLTM_Submission/bitcoin_dataset.csv) yang berisi data historis harga Bitcoin berbasis jam (*hourly*) dari tahun 2017 hingga 2023 dengan total lebih dari 53.000 baris data.

Fitur utama yang dianalisis:
- `Date` (Tanggal dan Waktu)
- `Close` (Harga Penutupan Bitcoin)
- `Volume` (Volume Transaksi)

---

## 🏗️ Arsitektur Model

### 1. Baseline LSTM
Model standar berbasis *Long Short-Term Memory* (LSTM) untuk menangkap pola temporal sekuensial sederhana sebelum melakukan proyeksi *multi-horizon*.

### 2. Seq2Seq LSTM Subclass (Autoregressive)
Model Sequence-to-Sequence yang diimplementasikan menggunakan teknik *subclassing* Keras (`keras.Model`). Model ini menggunakan pendekatan autoregresif di mana prediksi pada langkah sebelumnya digunakan sebagai input untuk prediksi langkah berikutnya selama proses inferensi.

### 3. Seq2Seq Functional
Model Sequence-to-Sequence yang dibangun menggunakan *Keras Functional API* dengan memisahkan bagian Encoder dan Decoder secara eksplisit untuk peramalan multi-step yang stabil.

---

## 📉 Fungsi Loss Kustom: Weighted Horizon Loss

Fungsi loss ini memberikan penalti lebih berat untuk kesalahan prediksi yang terjadi di masa depan yang lebih jauh. Bobot dihitung secara linier per *step* $i$:

$$\text{Bobot}_i = 1.0 + (i - 1) \times 0.1$$

Untuk $i \in [1, 24]$, bobot bergerak dari `1.0` hingga `3.3`.

---

## 🛠️ Cara Menjalankan Notebook

### Prasyarat
Pastikan Anda memiliki pustaka berikut terinstal di lingkungan Anda:
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

### Langkah Penggunaan
1. Clone repositori ini ke komputer lokal Anda:
   ```bash
   git clone https://github.com/Daapputra/Bitcoin-MultiHorizon-Forecasting.git
   ```
2. Jalankan Jupyter Notebook atau unggah ke Google Colab:
   - [Muhammad_Nur_Daffa_Naufal_Putra_Submission_Akhir_DLTM.ipynb](file:///c:/Users/majan/.gemini/antigravity-ide/scratch/DLTM_Submission/Muhammad_Nur_Daffa_Naufal_Putra_Submission_Akhir_DLTM.ipynb)
3. Pastikan file `bitcoin_dataset.csv` berada di direktori yang sama dengan notebook.
4. Jalankan sel-sel notebook secara berurutan.

---

## 📝 Hasil Evaluasi (MAE)

Evaluasi model dilakukan menggunakan metrik *Mean Absolute Error* (MAE) pada data pengujian untuk melihat perbandingan performa antara Baseline LSTM, Seq2Seq Subclass, dan Seq2Seq Functional. Model terbaik akan disimpan secara otomatis dengan format `.keras`.

---
*Dibuat oleh **Muhammad Nur Daffa Naufal Putra** Submission DLTM.*
