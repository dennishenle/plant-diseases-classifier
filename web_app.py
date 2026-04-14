"""
Flask web application for plant disease classification.

Loads the trained model once at startup and exposes:
  GET  /           -> renders templates/index.html
  POST /classify   -> accepts an uploaded image, returns JSON prediction

Usage:
    uv run web_app.py
"""

import io
import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from plant_diseases.device import select_device
from plant_diseases.model import PlantDiseaseModel
from plant_diseases.transforms import build_val_transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = Path("output/best_model.pt")

# ---------------------------------------------------------------------------
# Fail-fast model loading at module level
# ---------------------------------------------------------------------------

if not MODEL_PATH.exists():
    raise SystemExit(
        f"Model checkpoint not found: {MODEL_PATH}\n"
        "Train a model first with: uv run train.py"
    )

_device = select_device()
_model, _checkpoint = PlantDiseaseModel.from_checkpoint(MODEL_PATH, _device)
_classes: list[str] = _checkpoint.classes
_preprocess = build_val_transforms()

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit


@app.route("/", methods=["GET"])
def index():
    """Render the main upload page."""
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    """
    Accept an uploaded image and return the predicted disease label.

    Expects multipart/form-data with a file field named ``image``.
    Returns JSON: {"label": "...", "confidence": <float>}
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image file selected"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({"error": "Uploaded file is not a valid image"}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to open image: {exc}"}), 400

    label, confidence = _run_inference(image)
    return jsonify({"label": label, "confidence": confidence})


@torch.no_grad()
def _run_inference(image: Image.Image) -> tuple[str, float]:
    """Run a PIL image through the model and return (label, confidence)."""
    tensor = _preprocess(image).unsqueeze(0).to(_device)
    logits = _model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()
    return _classes[idx], probs[idx].item()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Using device: {_device}")
    print(f"Loaded {len(_classes)} classes from {MODEL_PATH}")
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
