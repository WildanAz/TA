import requests

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
