
# ==============================================================================
# File: prediksi.py
# Deskripsi: Berisi logika inti untuk prediksi menggunakan model dua tahap.
# ==============================================================================
import numpy as np
# Import semua variabel dan model yang dibutuhkan dari panggil_model.py
from panggil_model import (
    model_sehat_sakit, scaler_sehat_sakit, class_labels_model1,
    model_penyakit, scaler_penyakit, class_labels_model2,
    warna_kategori, tekstur_kategori, lokasi_kategori
)

def one_hot_encode(value, categories):
    """Fungsi untuk mengubah nilai kategorikal menjadi one-hot encoding."""
    one_hot = [0] * len(categories)
    if value in categories:
        one_hot[categories.index(value)] = 1
    return one_hot

def prediksi_klasifikasi(warna, tekstur, lokasi, luka):
    """
    Fungsi prediksi utama dengan arsitektur dua tahap.
    Mengembalikan label prediksi dan skor keyakinan (confidence).
    """
    # --- TAHAP 1: Prediksi Sehat vs. Sakit menggunakan Model 1 ---
    
    # 1. Preprocessing fitur untuk Model 1
    # 'tidak luka' menjadi 0, selain itu (luka, tidak terdeteksi, error) menjadi 1
    luka_bin = 0 if luka == 'tidakluka' else 1
    
    warna_oh = one_hot_encode(warna, warna_kategori)
    tekstur_oh = one_hot_encode(tekstur, tekstur_kategori)
    lokasi_oh = one_hot_encode(lokasi, lokasi_kategori)

    # Susun fitur sesuai urutan saat pelatihan Model 1
    # Urutan: [Luka, Warna(3), Tekstur(2), Lokasi(3)]
    # Pastikan urutan ini sama dengan urutan kolom di file CSV Anda
    input_data_model1 = np.array([
        [luka_bin] + warna_oh + tekstur_oh + lokasi_oh
    ])
    
    # 2. Scaling dan Prediksi dengan Model 1
    input_scaled_model1 = scaler_sehat_sakit.transform(input_data_model1)
    # Gunakan .predict() karena kita hanya butuh labelnya di tahap ini
    pred_idx_model1 = model_sehat_sakit.predict(input_scaled_model1)[0]
    pred_label_model1 = class_labels_model1[pred_idx_model1]

    # 3. Logika Kondisional
    if pred_label_model1 == 'sehat':
        # Jika diprediksi sehat, langsung kembalikan hasilnya. Selesai.
        probas = model_sehat_sakit.predict_proba(input_scaled_model1)[0]
        confidence = probas[pred_idx_model1]
        return 'sehat', confidence
    
    else: # Jika pred_label_model1 == 'sakit'
        # --- TAHAP 2: Prediksi jenis penyakit menggunakan Model 2 ---

        # 1. Preprocessing fitur untuk Model 2 (TANPA 'luka')
        # Urutan: [Warna(3), Tekstur(2), Lokasi(3)]
        input_data_model2 = np.array([
            warna_oh + tekstur_oh + lokasi_oh
        ])
        
        # 2. Scaling dan Prediksi dengan Model 2
        input_scaled_model2 = scaler_penyakit.transform(input_data_model2)
        
        # Ambil probabilitas untuk mendapatkan keyakinan model
        probas_model2 = model_penyakit.predict_proba(input_scaled_model2)[0]
        pred_idx_model2 = np.argmax(probas_model2)
        
        pred_label_model2 = class_labels_model2[pred_idx_model2]
        confidence = probas_model2[pred_idx_model2]
        
        # Kembalikan hasil dari model spesialis penyakit
        return pred_label_model2, confidence