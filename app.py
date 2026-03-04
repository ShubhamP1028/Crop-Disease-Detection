# app.py
# Flask app for inference on PlantVillage-trained Keras model (.h5)
# Single-file app.
import os
import io
import json
from typing import Any, Dict

from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
# Import TensorFlow/Keras
try:
    from tensorflow.keras.models import load_model as load_keras_model
    import tensorflow as tf
    HAS_KERAS = True
except Exception:
    HAS_KERAS = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration: paths (adjust or set environment variables)
KERAS_MODEL_PATH = os.environ.get("KERAS_MODEL_PATH", "crop_disease_model.h5")
LABEL_MAP_PATH = os.environ.get("LABEL_MAP_PATH", "leaf-map.json")
TREATMENT_DATA_PATH = os.environ.get("TREATMENT_DATA_PATH", "treatment-data.json")
TARGET_SIZE = (224, 224)  # expected input image size for the model

# Load label map (index -> classname)
label_map = None
if os.path.exists(LABEL_MAP_PATH):
    try:
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
            # If file maps names to ids, convert to id->name
            if all(isinstance(k, str) and k.isdigit() for k in label_map.keys()):
                label_map = {int(k): v for k, v in label_map.items()}
    except Exception:
        label_map = None

# Fallback: will set placeholder labels if none loaded
if label_map is None:
    # Create a simple label map (you should replace this with your actual classes)
    label_map = {i: f"class_{i}" for i in range(38)}  # PlantVillage has 38 classes

# Load Keras model
model = None
model_type = None
_resolved_model_path = os.path.abspath(KERAS_MODEL_PATH)
if HAS_KERAS and os.path.exists(_resolved_model_path):
    try:
        model = load_keras_model(_resolved_model_path)
        model_type = "keras"
        print(f"Loaded Keras model from {_resolved_model_path}")
    except Exception as e:
        print(f"Failed to load Keras model from {_resolved_model_path}: {e}")
else:
    print(
        "Keras model not loaded. "
        f"HAS_KERAS={HAS_KERAS} model_path='{_resolved_model_path}' exists={os.path.exists(_resolved_model_path)}"
    )


# Lazy-loaded treatment data cache
_treatment_data_cache = None


def load_treatment_data() -> Dict[str, Dict[str, Any]]:
    """
    Load treatment data from JSON.

    Structure:
    {
        "Disease_Label": {
            "disease": "...",
            "steps": [...],
            "prevention": [...],
            "expertNote": "...",
            "severity": "Low|Medium|High"
        },
        ...
    }
    """
    global _treatment_data_cache

    if _treatment_data_cache is not None:
        return _treatment_data_cache

    if not os.path.exists(TREATMENT_DATA_PATH):
        # No file available – start with empty mapping
        _treatment_data_cache = {}
        return _treatment_data_cache

    try:
        with open(TREATMENT_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Expect a dict keyed by disease label
            if isinstance(data, dict):
                _treatment_data_cache = data
            else:
                # Defensive: unexpected structure
                _treatment_data_cache = {}
    except Exception as e:
        print(f"Failed to load treatment data: {e}")
        _treatment_data_cache = {}

    return _treatment_data_cache

def preprocess_image_pil(image: Image.Image, target_size=TARGET_SIZE):
    """Convert PIL image to preprocessed numpy array for Keras models."""
    image = image.convert("RGB")
    image = image.resize(target_size, Image.BILINEAR)
    arr = np.array(image).astype("float32") / 255.0
    # shape: (H, W, C) -> (1, H, W, C)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(image: Image.Image):
    """Return (pred_label, confidence, all_scores)
    pred_label: string label
    confidence: float in [0,1]
    all_scores: list of (label, score) sorted desc
    """
    if model is None:
        return "No model loaded", 0.0, []
    
    x = preprocess_image_pil(image)
    preds = model.predict(x)[0]  # shape (num_classes,)
    
    # softmax if not already probabilities
    if preds.sum() <= 1.0 + 1e-6 and np.all(preds >= 0):
        probs = preds
    else:
        exp = np.exp(preds - np.max(preds))
        probs = exp / exp.sum()
    
    top_idx = int(np.argmax(probs))
    label = label_map.get(top_idx, str(top_idx))
    confidence = float(probs[top_idx])
    all_scores = [(label_map.get(i, str(i)), float(probs[i])) for i in range(len(probs))]
    all_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
    
    return label, confidence, all_scores

@app.route('/', methods=['GET'])
def index():
    # Serve the SPA HTML file (modern glassmorphism UI)
    return send_from_directory(os.path.dirname(__file__), 'index.html')


@app.route('/styles.css', methods=['GET'])
def styles_css():
    # Serve SPA stylesheet
    return send_from_directory(os.path.dirname(__file__), 'styles.css')


@app.route('/app.js', methods=['GET'])
def app_js():
    # Serve SPA JavaScript
    return send_from_directory(os.path.dirname(__file__), 'app.js')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded on server.",
            "model_path": _resolved_model_path,
            "has_keras": HAS_KERAS
        }), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for upload."}), 400
    
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        print(f"Error reading image: {e}")
        return jsonify({"error": "Invalid image file."}), 400

    try:
        label, confidence, scores = predict_image(image)

        # Encode image to display
        buffered = io.BytesIO()
        image.thumbnail((800, 800))
        image.save(buffered, format="PNG")
        import base64  # Local import to avoid issues if unused in other contexts
        image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        response_payload = {
            "label": label,
            "confidence": confidence,
            "scores": scores[:10],  # top 10 predictions
            "image_b64": image_b64,
        }

        return jsonify(response_payload), 200

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": "Error processing image."}), 500


@app.route('/treatment/<path:disease>', methods=['GET'])
def treatment(disease: str):
    """
    Return treatment information for a given disease label.

    The disease parameter is expected to match the model's label
    (e.g. "Tomato___Late_blight").
    """
    try:
        data = load_treatment_data()
    except Exception as e:
        # File read or parse error – surface as 500
        print(f"Error loading treatment data: {e}")
        return jsonify({"error": "Failed to load treatment data."}), 500

    # Try exact match first
    treatment_info = data.get(disease)

    # Fallback: try case-insensitive match on keys
    if treatment_info is None:
        lowered = {k.lower(): v for k, v in data.items()}
        treatment_info = lowered.get(disease.lower())

    # If still not found, return a sensible default treatment object
    if treatment_info is None:
        treatment_info = {
            "disease": disease,
            "steps": [
                "Remove and safely dispose of heavily affected leaves.",
                "Apply a recommended fungicide or bactericide where appropriate.",
                "Avoid overhead watering to keep foliage as dry as possible.",
                "Improve airflow by pruning crowded branches or plants.",
            ],
            "prevention": [
                "Use certified, disease-free seeds or seedlings.",
                "Rotate crops regularly and avoid planting the same crop in the same spot every season.",
                "Maintain proper plant spacing for good air circulation.",
                "Regularly monitor plants for early signs of stress or infection.",
            ],
            "expertNote": (
                "Specific treatment data for this disease was not found, "
                "so general best-practice plant protection advice is provided. "
                "Consult a local agronomist for region-specific recommendations."
            ),
            "severity": "Medium",
        }

    return jsonify(treatment_info), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "model_loaded": model is not None,
        "model_type": model_type,
        "num_labels": len(label_map),
        "model_path": _resolved_model_path,
        "has_keras": HAS_KERAS
    })

if __name__ == '__main__':
    # Debug mode by default for local testing
    app.run(host='0.0.0.0', port=5030, debug=True)