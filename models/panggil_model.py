# ==============================================================================
# File: panggil_model.py
# Deskripsi: Memuat semua model dan scaler yang sudah dilatih.
# ==============================================================================
import joblib

# --- Muat Aset untuk Model 1 (Sehat vs. Sakit) ---
model_sehat_sakit = joblib.load("model_sehat_sakit.pkl")
scaler_sehat_sakit = joblib.load("scaler_sehat_sakit.pkl")
# Label: 0 untuk Sehat, 1 untuk Sakit
class_labels_model1 = ['sehat', 'sakit'] 

# --- Muat Aset untuk Model 2 (Spesialis Penyakit) ---
model_penyakit = joblib.load("model_penyakit.pkl")
scaler_penyakit = joblib.load("scaler_penyakit.pkl")
# Label: 0, 1, 2 (sesuai urutan saat pelatihan)
class_labels_model2 = ['foot_rot', 'necrotic_stomatitis', 'pmk']

# --- Konfigurasi Kategori Fitur (untuk one-hot encoding) ---
# Urutan harus sama persis dengan saat pelatihan
warna_kategori = ['hitam', 'kuning', 'merah']
tekstur_kategori = ['halus', 'kasar']
lokasi_kategori = ['gusi', 'kuku', 'lidah']