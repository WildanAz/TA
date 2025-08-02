import streamlit as st
import requests
import numpy as np
import joblib

# ====== Setup ======
model = joblib.load("SVM_linear.pkl")
scaler = joblib.load("scaler.pkl")

warna_kategori = ['hitam', 'kuning', 'merah']
tekstur_kategori = ['halus', 'kasar']
lokasi_kategori = ['gusi', 'kuku', 'lidah']

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

def one_hot_encode(value, categories):
    one_hot = [0] * len(categories)
    if value in categories:
        idx = categories.index(value)
        one_hot[idx] = 1
    return one_hot

def prediksi_klasifikasi(warna, tekstur, lokasi, luka):
    warna_oh = one_hot_encode(warna, warna_kategori)
    tekstur_oh = one_hot_encode(tekstur, tekstur_kategori)
    lokasi_oh = one_hot_encode(lokasi, lokasi_kategori)
    luka_bin = 1 if luka == 'ya' else 0

    input_data = np.array([[luka_bin] + warna_oh + tekstur_oh + lokasi_oh])
    input_scaled = scaler.transform(input_data)
    class_labels = ['Foot_rot', 'necrotic_stomatitis', 'pmk', 'sehat']
    pred = model.predict(input_scaled)[0]
    return class_labels[pred]

def deteksi_fitur(img_bytes):
    prediksi = {}
    for fitur in urls:
        response = requests.post(urls[fitur], headers=headers[fitur], data=img_bytes)
        result = response.json()
        top = max(result["predictions"], key=lambda x: x["probability"])
        prediksi[fitur] = top["tagName"].lower().replace("_", "")
    return prediksi

# ========== STREAMLIT UI ==========

st.set_page_config(layout="wide", page_title="Prediksi Penyakit Sapi")

st.markdown("""
    <style>
    .stButton>button {
        background-color: #007BFF;
        color: white;
        padding: 0.5em 2em;
        margin: 0.5em;
        border-radius: 6px;
        font-weight: bold;
    }
    .box {

        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🐄 Aplikasi Prediksi Penyakit Sapi Berdasarkan Citra Luka")

col1, col2 = st.columns([1.5, 1.2])

with col1:
    uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="[Gambar]", use_column_width=True)
    pred_button = st.button("Prediksi")

with col2:
    if pred_button and uploaded_file:
        image_bytes = uploaded_file.read()
        fitur = deteksi_fitur(image_bytes)
        warna = fitur["warna"]
        tekstur = fitur["tekstur"]
        lokasi = fitur["lokasi"]
        luka = "ya" if fitur["luka"] == "luka" else "tidak"

        penyakit = prediksi_klasifikasi(warna, tekstur, lokasi, luka)
        saran = penanganan_dict[penyakit]

        st.markdown("<div class='box'><b>Karakteristik Citra</b>", unsafe_allow_html=True)
        st.write(f"Lokasi: {lokasi}")
        st.write(f"Luka: {luka}")
        st.write(f"Warna: {warna}")
        st.write(f"Tekstur: {tekstur}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='box'><b>Prediksi</b>", unsafe_allow_html=True)
        st.write(f"Penyakit: {penyakit.upper()}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='box'><b>Rekomendasi Penanganan</b>", unsafe_allow_html=True)
        for i, langkah in enumerate(saran, 1):
            st.write(f"{i}. {langkah}")
        st.markdown("</div>", unsafe_allow_html=True)

    elif pred_button:
        st.warning("Silakan unggah gambar terlebih dahulu.")
