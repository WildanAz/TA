import joblib

model = joblib.load("SVM.pkl")
scaler = joblib.load("scaler.pkl")

warna_kategori = ['hitam', 'kuning', 'merah']
tekstur_kategori = ['halus', 'kasar']
lokasi_kategori = ['gusi', 'kuku', 'lidah']
class_labels = ['Foot_rot', 'necrotic_stomatitis', 'pmk', 'sehat']
