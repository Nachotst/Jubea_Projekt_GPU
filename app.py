"""Lokale Website für den Jubaea-Bildklassifikator."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

from grad_cam import GradCAM, overlay_heatmap

classes = ["Not_Jubaea", "Jubaea"]
class_labels = {
    "Jubaea": "Jubaea chilensis",
    "Not_Jubaea": "Keine Jubaea",
}
image_size = 224
model_path = "Jubaea_with_val_resnet.pth"
allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
max_upload_bytes = 12 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = max_upload_bytes

transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


#GPU oder CPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def load_model():
    net = models.resnet18(weights=None)
    net.load_state_dict(torch.load('Jubaea_with_val_resnet.pth', map_location=device))
    net = net.to(device)
    net.eval()

    return net


model = load_model()
grad_cam = GradCAM(model, model.layer4[-1])


def _array_to_data_url(array: np.ndarray) -> str:
    pixels = np.uint8(np.clip(array, 0, 1) * 255)
    image = Image.fromarray(pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_name=Path(model_path).name,
        device=str(device),
    )


@app.route("/classify", methods=["POST"])
def classify():
    uploaded = request.files.get("image")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "Bitte ein Bild hochladen."}), 400

    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in allowed_extensions:
        return jsonify({"error": "Bitte ein Bild im Format JPG, PNG, WEBP oder BMP wählen."}), 400

    try:
        image = Image.open(uploaded.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Die Datei konnte nicht als Bild gelesen werden."}), 400

    input_tensor = transform(image).unsqueeze(0).to(device)
    cam, predicted_class, confidence = grad_cam.generate(input_tensor)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    _original, heatmap, overlay = overlay_heatmap(buffer, cam, image_size=image_size)

    class_name = classes[predicted_class]
    return jsonify(
        {
            "class_id": predicted_class,
            "class_name": class_name,
            "class_label": class_labels[class_name],
            "confidence": confidence,
            "heatmap": _array_to_data_url(heatmap),
            "overlay": _array_to_data_url(overlay),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
