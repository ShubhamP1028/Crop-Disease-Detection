# app.py
# Flask app for inference on PlantVillage-trained Keras model (.h5)
# Single-file app.
import os
import io
import json
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, jsonify
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
if HAS_KERAS and os.path.exists(KERAS_MODEL_PATH):
    try:
        model = load_keras_model(KERAS_MODEL_PATH)
        model_type = "keras"
        print(f"Loaded Keras model from {KERAS_MODEL_PATH}")
    except Exception as e:
        print("Failed to load Keras model:", e)
else:
    print("Keras model not loaded. Check your model path and Keras installation.")

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

import base64

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',
                          model_loaded=(model is not None),
                          model_type=model_type,
                          label_map=label_map,
                          result=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        label, confidence, scores = predict_image(image)
        
        # Encode image to display
        buffered = io.BytesIO()
        image.thumbnail((800, 800))
        image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Create result object for template
        result = {
            'label': label,
            'confidence': confidence,
            'scores': scores[:10]  # Show top 10 predictions
        }
        
        return render_template('index.html',
                              model_loaded=(model is not None),
                              model_type=model_type,
                              label_map=label_map,
                              result=result,
                              image_b64=image_b64)
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return render_template('index.html',
                              model_loaded=(model is not None),
                              model_type=model_type,
                              label_map=label_map,
                              result=None,
                              error=str(e))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "model_loaded": model is not None,
        "model_type": model_type,
        "num_labels": len(label_map)
    })

if __name__ == '__main__':
    # Debug mode by default for local testing
    app.run(host='0.0.0.0', port=5030, debug=True)