from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

API_KEY = "mysecret123"

model = YOLO("best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if request.headers.get("api-key") != API_KEY:
        return {"error": "Unauthorized"}, 403

    if 'image' not in request.files:
        return {"error": "No image provided"}, 400

    file = request.files['image']

    filepath = "temp.jpg"
    file.save(filepath)

    results = model(filepath)

    detections = []

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            detections.append(label)

    return jsonify({"detections": detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    