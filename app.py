import os
import numpy as np
import joblib
import requests
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

# --- 1. INISIALISASI APLIKASI FLASK ---
app = Flask(__name__)

# --- 2. KONFIGURASI APLIKASI & KUNCI API ---
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

AZURE_PREDICTION_KEY = "3n6ucLHlVi6XLXa606X3oQvxMBJ31rWugpNS0L106mWENLGJ3qKEJQQJ99BDACYeBjFXJ3w3AAAIACOG7P6D"
# --- 3. PEMUATAN MODEL DAN SCALER ---
try:
    # Model 1 tidak lagi digunakan dalam logika utama, tapi tetap dimuat jika diperlukan
    model_sehat_sakit = joblib.load("model_sehat_sakit.pkl")
    scaler_sehat_sakit = joblib.load("scaler_sehat_sakit.pkl")
    
    # Model 2 adalah model utama untuk diagnosis penyakit
    model_penyakit = joblib.load("model_penyakit.pkl")
    scaler_penyakit = joblib.load("scaler_penyakit.pkl")
    class_labels_model2 = ['foot_rot', 'necrotic_stomatitis', 'pmk']
except FileNotFoundError as e:
    print(f"Error: File model tidak ditemukan. Pastikan file .pkl ada di folder yang sama.")
    print(e)
    # exit()

# --- 4. KONFIGURASI FITUR & DATA ---
warna_kategori = ['hitam', 'kuning', 'merah']
tekstur_kategori = ['halus', 'kasar']
lokasi_kategori = ['gusi', 'kuku', 'lidah']

AZURE_URLS = {
    "lokasi": "https://willcustommodel-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/fbdee0e8-0a8f-4269-91ba-942e968805d0/classify/iterations/Iteration3/image",
    "warna": "https://willcustommodel-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/fbdee0e8-0a8f-4269-91ba-942e968805d0/classify/iterations/Iteration7/image",
    "tekstur": "https://willcustommodel-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/fbdee0e8-0a8f-4269-91ba-942e968805d0/classify/iterations/Iteration6/image",
    "luka": "https://willcustommodel-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/fbdee0e8-0a8f-4269-91ba-942e968805d0/classify/iterations/Iteration4/image"
}

AZURE_HEADERS = {
    'Content-Type': 'application/octet-stream',
    'Prediction-Key': AZURE_PREDICTION_KEY
}

PENANGANAN_DICT = {
    'pmk': [
        "Isolasi sapi yang terinfeksi untuk mencegah penyebaran lebih lanjut.",
        "Pemberian antibiotik untuk mencegah infeksi sekunder serta analgesik untuk mengurangi rasa sakit.",
        "Vaksinasi PMK sebagai langkah pencegahan utama, terutama di wilayah endemik."
    ],
    'foot_rot': [
        "Pembersihan luka dan aplikasi antibiotik topikal untuk menghambat pertumbuhan bakteri.",
        "Pemberian antibiotik sistemik, dalam kasus infeksi yang lebih parah.",
        "Peningkatan sanitasi kandang, terutama menjaga lantai tetap kering dan bersih."
    ],
    'necrotic_stomatitis': [
        "Pemberian antibiotik sistemik untuk menghambat pertumbuhan bakteri.",
        "Perawatan luka dengan antiseptik oral untuk mempercepat penyembuhan.",
        "Peningkatan sanitasi pakan dan air minum untuk mencegah infeksi ulang."
    ],
    'sehat': ["Tidak perlu penanganan khusus."]
}

# --- 5. Preprocessing
def one_hot_encode(value, categories):
    one_hot = [0] * len(categories)
    if value in categories:
        one_hot[categories.index(value)] = 1
    return one_hot

def prediksi_klasifikasi(warna, tekstur, lokasi, luka):
    """
    Fungsi prediksi dengan logika bisnis yang diperkuat dan aturan pasca-proses.
    """
    is_luka = luka != 'tidakluka'

    if not is_luka:
        return 'sehat', 1.0
    else:
        warna_oh = one_hot_encode(warna, warna_kategori)
        tekstur_oh = one_hot_encode(tekstur, tekstur_kategori)
        lokasi_oh = one_hot_encode(lokasi, lokasi_kategori)
        
        input_data_model2 = np.array([warna_oh + tekstur_oh + lokasi_oh])
        input_scaled_model2 = scaler_penyakit.transform(input_data_model2)
        probas_model2 = model_penyakit.predict_proba(input_scaled_model2)[0]
        
        initial_pred_idx = np.argmax(probas_model2)
        pred_label = class_labels_model2[initial_pred_idx]
        confidence = probas_model2[initial_pred_idx]
        
        if pred_label == 'necrotic_stomatitis' and warna == 'merah':
            sorted_indices = np.argsort(probas_model2)[::-1]
            for idx in sorted_indices:
                if class_labels_model2[idx] != 'necrotic_stomatitis':
                    pred_label = class_labels_model2[idx]
                    confidence = probas_model2[idx]
                    break 

        elif pred_label == 'foot_rot' and (lokasi == 'gusi' or lokasi == 'lidah'):
            sorted_indices = np.argsort(probas_model2)[::-1]
            for idx in sorted_indices:
                if class_labels_model2[idx] != 'foot_rot':
                    pred_label = class_labels_model2[idx]
                    confidence = probas_model2[idx]
                    break
        
        elif pred_label == 'pmk' and (warna == 'hitam' or warna == 'kuning'):
            sorted_indices = np.argsort(probas_model2)[::-1]
            for idx in sorted_indices:
                if class_labels_model2[idx] != 'pmk':
                    pred_label = class_labels_model2[idx]
                    confidence = probas_model2[idx]
                    break
        

        elif pred_label == 'foot_rot' and warna == 'merah':
            sorted_indices = np.argsort(probas_model2)[::-1]
            for idx in sorted_indices:
                if class_labels_model2[idx] != 'foot_rot':
                    pred_label = class_labels_model2[idx]
                    confidence = probas_model2[idx]
                    break

        return pred_label, confidence

def deteksi_fitur_azure(img_bytes):
    hasil_fitur, confidence_fitur = {}, {}
    for fitur, url in AZURE_URLS.items():
        try:
            res = requests.post(url, headers=AZURE_HEADERS, data=img_bytes)
            res.raise_for_status()
            prediksi = res.json()["predictions"]
            top_prediction = max(prediksi, key=lambda x: x["probability"]) if prediksi else {"tagName": "Tidak Terdeteksi", "probability": 0.0}
            hasil_fitur[fitur] = top_prediction["tagName"].lower().replace("_", "")
            confidence_fitur[fitur] = round(top_prediction["probability"] * 100, 2)
        except requests.exceptions.RequestException as e:
            print(f"Error saat menghubungi Azure API untuk fitur '{fitur}': {e}")
            hasil_fitur[fitur], confidence_fitur[fitur] = "error", 0
    return hasil_fitur, confidence_fitur

# --- 6. ROUTING APLIKASI ---
@app.route("/", methods=["GET", "POST"])
def index():
    result, image_url, error_message = None, None, None
    if request.method == "POST":
        if AZURE_PREDICTION_KEY == "MASUKKAN_PREDICTION_KEY_ANDA_DI_SINI":
            error_message = "Kunci Prediksi Azure belum diatur di file app.py!"
            return render_template("index.html", error_message=error_message)
        
        file = request.files.get("image")
        if not file or file.filename == '':
            error_message = "Silakan pilih file gambar untuk diunggah."
            return render_template("index.html", error_message=error_message)

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        image_url = url_for('static', filename=f'uploads/{filename}')

        with open(save_path, 'rb') as f:
            img_bytes = f.read()

        fitur, confidence = deteksi_fitur_azure(img_bytes)
        
        # --- Validasi Keyakinan Fitur ---
        fitur_rendah = []
        # Periksa semua fitur yang diekstrak
        for f in ['warna', 'tekstur', 'lokasi', 'luka']:
            if confidence[f] < 60:
                fitur_rendah.append(f"{f.title()} ({confidence[f]}%)")

        if fitur_rendah:
            # Jika ada fitur dengan keyakinan rendah, buat pesan error
            error_message = f"Keyakinan fitur terlalu rendah: {', '.join(fitur_rendah)}. Silakan unggah gambar yang lebih jelas atau relevan dengan kasus."
        else:
            # Jika semua fitur valid, lanjutkan ke prediksi
            hasil_diagnosis, keyakinan_diagnosis = prediksi_klasifikasi(
                warna=fitur['warna'], tekstur=fitur['tekstur'],
                lokasi=fitur['lokasi'], luka=fitur['luka']
            )
            rekomendasi = PENANGANAN_DICT.get(hasil_diagnosis, ["Informasi penanganan tidak tersedia."])
            result = {
                "fitur": fitur, "confidence": confidence,
                "diagnosis": hasil_diagnosis.replace('_', ' ').title(),
                "keyakinan_diagnosis": round(keyakinan_diagnosis * 100, 2),
                "rekomendasi": rekomendasi
            }
            
    return render_template("index.html", result=result, image_url=image_url, error_message=error_message)


if __name__ == "__main__":
    app.run(debug=True)
