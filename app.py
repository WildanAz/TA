from flask import Flask, render_template, request, url_for
<<<<<<< HEAD
import numpy as np
import joblib
import requests
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folder penyimpanan gambar sementara
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model dan scaler
model = joblib.load("SVM.pkl")
scaler = joblib.load("scaler.pkl")

# Kategori fitur
warna_kategori = ['hitam', 'kuning', 'merah']
tekstur_kategori = ['halus', 'kasar']
lokasi_kategori = ['gusi', 'kuku', 'lidah']
class_labels = ['Foot_rot', 'necrotic_stomatitis', 'pmk', 'sehat']

# Saran penanganan
=======
from werkzeug.utils import secure_filename
import os

from services.azure_predict import deteksi_fitur
from models.prediksi import prediksi_klasifikasi
from models.panggil_model import class_labels

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


>>>>>>> 4e4ab12 (update project)
penanganan_dict = {
    'pmk': [
        "Isolasi sapi yang terinfeksi untuk mencegah penyebaran lebih lanjut.",
        "Pemberian antibiotik untuk mencegah infeksi sekunder serta analgesik untuk mengurangi rasa sakit.",
        "Vaksinasi PMK sebagai langkah pencegahan utama, terutama di wilayah endemik."
    ],
    'Foot_rot': [
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

<<<<<<< HEAD
# Azure Prediction
urls = {
    "lokasi": "https://willcustommodel-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/fbdee0e8-0a8f-4269-91ba-942e968805d0/classify/iterations/Iteration3/image",
    "warna": "https://willcustommodel-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/fbdee0e8-0a8f-4269-91ba-942e968805d0/classify/iterations/Iteration7/image",
    "tekstur": "https://willcustommodel-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/fbdee0e8-0a8f-4269-91ba-942e968805d0/classify/iterations/Iteration6/image",
    "luka": "https://willcustommodel-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/fbdee0e8-0a8f-4269-91ba-942e968805d0/classify/iterations/Iteration4/image"
}
headers = {
    fitur: {
        'Content-Type': 'application/octet-stream',
        'Prediction-Key': '3n6ucLHlVi6XLXa606X3oQvxMBJ31rWugpNS0L106mWENLGJ3qKEJQQJ99BDACYeBjFXJ3w3AAAIACOG7P6D'
    } for fitur in urls
}

# Fungsi bantu
def one_hot_encode(value, categories):
    one_hot = [0] * len(categories)
    if value in categories:
        one_hot[categories.index(value)] = 1
    return one_hot

def deteksi_fitur(img_bytes):
    hasil, confidence = {}, {}
    for fitur in urls:
        res = requests.post(urls[fitur], headers=headers[fitur], data=img_bytes)
        pred = res.json()["predictions"]
        top = max(pred, key=lambda x: x["probability"])
        hasil[fitur] = top["tagName"].lower().replace("_", "")
        confidence[fitur] = {
            p["tagName"].lower().replace("_", ""): round(p["probability"] * 100, 2)
            for p in pred
        }
    return hasil, confidence

def prediksi_klasifikasi(warna, tekstur, lokasi, luka):
    warna_oh = one_hot_encode(warna, warna_kategori)
    tekstur_oh = one_hot_encode(tekstur, tekstur_kategori)
    lokasi_oh = one_hot_encode(lokasi, lokasi_kategori)
    luka_bin = 1 if luka == 'ya' else 0

    input_data = np.array([[luka_bin] + warna_oh + tekstur_oh + lokasi_oh])
    input_scaled = scaler.transform(input_data)
    probas = model.predict_proba(input_scaled)[0]
    pred_idx = np.argmax(probas)
    pred_label = class_labels[pred_idx]

    if luka_bin == 1 and pred_label == 'sehat':
        for i in np.argsort(probas)[::-1]:
            if class_labels[i] != 'sehat':
                pred_label = class_labels[i]
                break
    if luka_bin == 0 and pred_label != 'sehat':
        pred_label = 'sehat'

    if pred_label == 'necrotic_stomatitis' and (warna == 'merah' or lokasi == 'kuku'):
        for i in np.argsort(probas)[::-1]:
            if class_labels[i] != 'necrotic_stomatitis':
                pred_label = class_labels[i]
                break
    return pred_label

# Routing utama
=======
>>>>>>> 4e4ab12 (update project)
@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)
            image_url = url_for('static', filename=f'uploads/{filename}')

            with open(save_path, 'rb') as f:
                img_bytes = f.read()

            fitur, confidence = deteksi_fitur(img_bytes)

            warna = fitur['warna']
            tekstur = fitur['tekstur']
            lokasi = fitur['lokasi']
            luka = 'ya' if fitur['luka'] == 'luka' else 'tidak'

            hasil = prediksi_klasifikasi(warna, tekstur, lokasi, luka)
            rekomendasi = penanganan_dict[hasil]

            result = {
                "fitur": fitur,
                "confidence": confidence,
                "diagnosis": hasil,
                "rekomendasi": rekomendasi,
                "luka": luka
            }

    return render_template("index.html", result=result, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
