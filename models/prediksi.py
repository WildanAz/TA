import numpy as np
from .panggil_model import model, scaler, warna_kategori, tekstur_kategori, lokasi_kategori, class_labels
from utils.preprocessing import one_hot_encode

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

    if warna == 'hitam' and tekstur == 'kasar' and lokasi == 'kuku' and luka == 'ya':
        pred_label = 'Foot_rot'
    return pred_label
