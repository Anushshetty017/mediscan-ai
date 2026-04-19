"""
Flask backend for Medical Image Classification
Supports Brain Tumor MRI and Skin Cancer (ISIC) classification
using EfficientNet models.
"""

import os
import io
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
from flask import Flask, request, jsonify, render_template


# ==========================================
# APP SETUP
# ==========================================
app = Flask(__name__)

# ==========================================
# DEVICE CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on: {device}")

# ==========================================
# CLASS LABELS
# ==========================================
BRAIN_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SKIN_CLASSES = [
    "vascular lesion",
    "squamous cell carcinoma",
    "seborrheic keratosis",
    "pigmented benign keratosis",
    "nevus",
    "melanoma",
    "dermatofibroma",
    "basal cell carcinoma",
    "actinic keratosis"
]

# ==========================================
# MODEL PATHS
# ==========================================
BRAIN_MODEL_PATH = os.path.join("models", "brain_model.pth")
SKIN_MODEL_PATH  = os.path.join("models", "skin_model.pth")

# ==========================================
# MODEL BUILDER
# ==========================================
def build_efficientnet(num_classes: int) -> nn.Module:
    """
    Build an EfficientNet-B0 model with a custom classification head.
    """
    model = EfficientNet.from_pretrained("efficientnet-b0")
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    return model


def load_model(model_path: str, num_classes: int) -> nn.Module:
    """
    Load a saved EfficientNet model from disk.
    Returns None if the file doesn't exist.
    """
    if not os.path.exists(model_path):
        print(f"[WARNING] Model file not found: {model_path}")
        return None

    model = build_efficientnet(num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model from {model_path}")
    return model


# ==========================================
# LOAD BOTH MODELS AT STARTUP
# ==========================================
brain_model = load_model(BRAIN_MODEL_PATH, num_classes=len(BRAIN_CLASSES))
skin_model  = load_model(SKIN_MODEL_PATH,  num_classes=len(SKIN_CLASSES))

# ==========================================
# PREPROCESSING PIPELINES
# ==========================================
# Standard ImageNet normalization used during training
brain_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

skin_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================================
# PREDICTION HELPER
# ==========================================
def predict(image_bytes: bytes, model: nn.Module, transform, class_names: list) -> dict:
    """
    Run inference on raw image bytes.
    Returns predicted class name and confidence score.
    """
    # Open image and convert to RGB (handles grayscale & RGBA too)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess: resize, normalize, add batch dimension
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)                         # raw logits
        probs   = torch.softmax(outputs, dim=1)         # probabilities
        conf, pred_idx = torch.max(probs, dim=1)        # top prediction

    predicted_class = class_names[pred_idx.item()]
    confidence      = round(conf.item() * 100, 2)       # as percentage

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }

# ==========================================
# ROUTES
# ==========================================
@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    POST /predict
    Form data:
        - file       : image file
        - model_type : "brain" or "skin"
    Returns JSON with predicted_class and confidence.
    """

    # --- Validate inputs ---
    if "file" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    model_type = request.form.get("model_type", "").strip().lower()
    if model_type not in ("brain", "skin"):
        return jsonify({"error": "Invalid model_type. Choose 'brain' or 'skin'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename. Please select an image."}), 400

    # --- Route to correct model ---
    if model_type == "brain":
        if brain_model is None:
            return jsonify({"error": "Brain model not loaded. Place brain_model.pth in /models folder."}), 503
        result = predict(file.read(), brain_model, brain_transform, BRAIN_CLASSES)

    else:  # skin
        if skin_model is None:
            return jsonify({"error": "Skin model not loaded. Place skin_model.pth in /models folder."}), 503
        result = predict(file.read(), skin_model, skin_transform, SKIN_CLASSES)

    result["model_type"] = model_type
    return jsonify(result)


# ==========================================
# ENTRY POINT
# ==========================================
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)