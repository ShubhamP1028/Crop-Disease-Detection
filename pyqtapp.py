import sys
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QFileDialog, QTextEdit, QWidget,
                            QProgressBar, QMessageBox, QFrame, QGridLayout, QScrollArea, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QFont, QPalette, QColor, QIcon, QLinearGradient

class ModernButton(QPushButton):
    def __init__(self, text, icon=None, primary=False):
        super().__init__(text)
        self.setMinimumHeight(45)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.primary = primary
        self._setup_style()
        
    def _setup_style(self):
        if self.primary:
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #10b981, stop:1 #059669);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    font-weight: bold;
                    font-size: 14px;
                    padding: 12px 20px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #34d399, stop:1 #10b981);
                    transform: translateY(-1px);
                }
                QPushButton:pressed {
                    background: #059669;
                }
                QPushButton:disabled {
                    background: #6b7280;
                    color: #9ca3af;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #374151, stop:1 #4b5563);
                    color: white;
                    border: 2px solid #4b5563;
                    border-radius: 12px;
                    font-weight: bold;
                    font-size: 14px;
                    padding: 10px 18px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #4b5563, stop:1 #6b7280);
                    border: 2px solid #6b7280;
                }
                QPushButton:pressed {
                    background: #374151;
                }
                QPushButton:disabled {
                    background: #374151;
                    color: #6b7280;
                    border: 2px solid #374151;
                }
            """)

class GradientHeader(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(80)
        self.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                          stop:0 #059669, stop:0.5 #10b981, stop:1 #34d399);
                color: white;
                font-size: 28px;
                font-weight: bold;
                border-radius: 15px;
                margin: 10px;
            }
        """)

class CardWidget(QFrame):
    def __init__(self, title=None):
        super().__init__()
        self.setObjectName("card")
        self.setStyleSheet("""
            #card {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                          stop:0 #1f2937, stop:1 #111827);
                border: 1px solid #374151;
                border-radius: 15px;
                padding: 0px;
            }
        """)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)
        
        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet("""
                QLabel {
                    color: #10b981;
                    font-size: 18px;
                    font-weight: bold;
                    padding-bottom: 5px;
                    border-bottom: 2px solid #374151;
                }
            """)
            self.layout.addWidget(title_label)

class AnimatedProgressBar(QProgressBar):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(8)
        self.setTextVisible(False)
        self.setStyleSheet("""
            QProgressBar {
                background-color: #374151;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                          stop:0 #10b981, stop:1 #34d399);
                border-radius: 4px;
            }
        """)

class ModelLoader(QThread):
    """Thread for loading the model to prevent UI freezing"""
    finished = pyqtSignal(bool)
    error = pyqtSignal(str)
    
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.model = None
    
    def run(self):
        try:
            # Load the trained model
            self.model = tf.keras.models.load_model(self.model_path)
            self.finished.emit(True)
        except Exception as e:
            self.error.emit(f"Error loading model: {str(e)}")
            self.finished.emit(False)

class PredictionWorker(QThread):
    """Thread for running predictions to prevent UI freezing"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, model, image_path, class_names, target_size, external_rescale):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.class_names = class_names
        self.target_size = target_size
        self.external_rescale = external_rescale
    
    def preprocess_image(self, image_path):
        """Preprocess image for model prediction"""
        try:
            # Load and preprocess the image
            image = Image.open(image_path)
            image = image.convert('RGB')
            image = image.resize(self.target_size)
            
            # Convert to numpy array and normalize
            image_array = np.array(image)
            image_array = image_array.astype('float32')
            # Avoid double-rescaling if the model already contains a Rescaling layer
            if self.external_rescale:
                image_array = image_array / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def run(self):
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(self.image_path)
            
            # Make prediction
            raw_preds = self.model.predict(processed_image)[0]
            # Match web app behavior: use probabilities if they already sum to ~1 and are non-negative, else softmax
            if raw_preds.sum() <= 1.0 + 1e-6 and np.all(raw_preds >= 0):
                probs = raw_preds
            else:
                exp = np.exp(raw_preds - np.max(raw_preds))
                probs = exp / exp.sum()

            predicted_class_index = int(np.argmax(probs))
            confidence = float(probs[predicted_class_index])

            # Build scores list sorted desc (label, score)
            scores = [(self.class_names[i] if i < len(self.class_names) else f"Class_{i}", float(probs[i])) for i in range(len(probs))]
            scores = sorted(scores, key=lambda x: x[1], reverse=True)

            # Top N (align with web default top 10, we'll present top 10 in UI)
            top_predictions = [
                {
                    'rank': i + 1,
                    'class': label,
                    'confidence': score,
                    'percentage': score * 100
                }
                for i, (label, score) in enumerate(scores[:10])
            ]

            result = {
                'label': self.class_names[predicted_class_index] if predicted_class_index < len(self.class_names) else f"Class_{predicted_class_index}",
                'confidence': confidence,
                'percentage': confidence * 100,
                'scores': scores,
                'top_predictions': top_predictions
            }
            
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(f"Error during prediction: {str(e)}")

class CropDiseaseDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_image_path = None
        self.model_input_size = (224, 224)
        self.external_rescale = True
        
        # Load class names from mapping if available, else fallback
        self.class_names = self.load_class_names()
        
        self.init_ui()
        self.apply_styles()
        
        # Optionally auto-load default model if present
        default_model_path = os.path.join(os.getcwd(), "crop_disease_model.h5")
        if os.path.exists(default_model_path):
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.model_loader = ModelLoader(default_model_path)
            self.model_loader.finished.connect(self.on_model_loaded)
            self.model_loader.error.connect(self.on_model_error)
            self.model_loader.start()
        
    def init_ui(self):
        """Initialize the modern user interface"""
        self.setWindowTitle("🌿 PlantGuard AI - Crop Disease Detection")
        self.setGeometry(100, 50, 1200, 800)
        self.setMinimumSize(1000, 700)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Left panel for controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel for image display and results
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_left_panel(self):
        """Create modern left control panel"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)
        
        # App header with gradient
        header = GradientHeader("🌿 PlantGuard AI")
        left_layout.addWidget(header)
        
        # Model card
        model_card = CardWidget("🤖 AI Model")
        model_layout = QVBoxLayout()
        
        self.load_model_btn = ModernButton("📁 Load AI Model", primary=True)
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)
        
        self.model_status_label = QLabel("⚪ No model loaded")
        self.model_status_label.setStyleSheet("color: #9ca3af; font-size: 13px;")
        model_layout.addWidget(self.model_status_label)
        
        model_card.layout.addLayout(model_layout)
        left_layout.addWidget(model_card)
        
        # Image card
        image_card = CardWidget("🖼️ Plant Image")
        image_layout = QVBoxLayout()
        
        self.select_image_btn = ModernButton("📷 Select Plant Image")
        self.select_image_btn.clicked.connect(self.select_image)
        self.select_image_btn.setEnabled(False)
        image_layout.addWidget(self.select_image_btn)
        
        self.image_status_label = QLabel("⚪ No image selected")
        self.image_status_label.setStyleSheet("color: #9ca3af; font-size: 13px;")
        image_layout.addWidget(self.image_status_label)
        
        image_card.layout.addLayout(image_layout)
        left_layout.addWidget(image_card)
        
        # Analysis card
        analysis_card = CardWidget("🔍 Analysis")
        analysis_layout = QVBoxLayout()
        
        self.predict_btn = ModernButton("🚀 Analyze Plant Health", primary=True)
        self.predict_btn.clicked.connect(self.predict_disease)
        self.predict_btn.setEnabled(False)
        analysis_layout.addWidget(self.predict_btn)
        
        self.progress_bar = AnimatedProgressBar()
        self.progress_bar.setVisible(False)
        analysis_layout.addWidget(self.progress_bar)
        
        analysis_card.layout.addLayout(analysis_layout)
        left_layout.addWidget(analysis_card)
        
        # Stats card
        stats_card = CardWidget("📈 Quick Stats")
        stats_layout = QVBoxLayout()
        
        stats_text = QLabel("""
        <style>
        .stat-item { margin: 8px 0; color: #d1d5db; }
        .stat-value { color: #10b981; font-weight: bold; }
        </style>
        <div class="stat-item">🌱 Supported Plants: <span class="stat-value">15+</span></div>
        <div class="stat-item">🔬 Diseases Detected: <span class="stat-value">38+</span></div>
        <div class="stat-item">⚡ Analysis Speed: <span class="stat-value">~2s</span></div>
        <div class="stat-item">🎯 Accuracy: <span class="stat-value">Up to 95%</span></div>
        """)
        stats_text.setTextFormat(Qt.TextFormat.RichText)
        stats_layout.addWidget(stats_text)
        
        stats_card.layout.addLayout(stats_layout)
        left_layout.addWidget(stats_card)
        
        left_layout.addStretch()
        return left_widget
    
    def create_right_panel(self):
        """Create modern right display panel"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)
        
        # Image display card
        image_display_card = CardWidget("🖼️ Plant Image Preview")
        image_display_layout = QVBoxLayout()
        
        # Image container with modern styling
        image_container = QFrame()
        image_container.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                          stop:0 #111827, stop:1 #1f2937);
                border: 2px dashed #374151;
                border-radius: 12px;
            }
        """)
        image_container_layout = QVBoxLayout(image_container)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("""
            QLabel {
                color: #6b7280;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.image_label.setText("🖼️\n\nNo image loaded\n\nSelect an image to begin analysis")
        image_container_layout.addWidget(self.image_label)
        
        image_display_layout.addWidget(image_container)
        image_display_card.layout.addLayout(image_display_layout)
        right_layout.addWidget(image_display_card)
        
        # Results card
        results_card = CardWidget("📊 Analysis Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(250)
        self.results_text.setPlaceholderText("🔍 Analysis results will appear here...")
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #111827;
                border: 1px solid #374151;
                border-radius: 10px;
                padding: 15px;
                color: #e5e7eb;
                font-family: 'Segoe UI', system-ui;
                font-size: 14px;
                line-height: 1.4;
            }
        """)
        results_layout.addWidget(self.results_text)
        
        results_card.layout.addLayout(results_layout)
        right_layout.addWidget(results_card)
        
        return right_widget
    
    def apply_styles(self):
        """Apply modern gradient styling"""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                          stop:0 #0f172a, stop:0.5 #1e293b, stop:1 #334155);
                color: #f8fafc;
            }
            QWidget {
                background: transparent;
            }
        """)
    
    def load_model(self):
        """Load the trained model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select AI Model File",
            "",
            "Keras Model Files (*.h5);;All Files (*)"
        )
        
        if file_path:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            
            # Load model in separate thread
            self.model_loader = ModelLoader(file_path)
            self.model_loader.finished.connect(self.on_model_loaded)
            self.model_loader.error.connect(self.on_model_error)
            self.model_loader.start()
    
    def on_model_loaded(self, success):
        """Handle model loading completion"""
        self.progress_bar.setVisible(False)
        
        if success:
            self.model = self.model_loader.model
            self.model_status_label.setText("🟢 AI Model Ready")
            self.model_status_label.setStyleSheet("color: #10b981; font-weight: bold; font-size: 13px;")
            self.select_image_btn.setEnabled(True)
            # Infer model input size and preprocessing behavior
            self.model_input_size = self.get_model_input_size(self.model)
            self.external_rescale = self.detect_external_rescale(self.model)
            
            # Update UI
            QMessageBox.information(self, "Success", "🤖 AI Model loaded successfully!\n\nYou can now select plant images for analysis.")
        else:
            self.model_status_label.setText("🔴 Failed to load model")
            self.model_status_label.setStyleSheet("color: #ef4444; font-size: 13px;")
    
    def on_model_error(self, error_message):
        """Handle model loading error"""
        self.progress_bar.setVisible(False)
        self.model_status_label.setText("🔴 Model Error")
        self.model_status_label.setStyleSheet("color: #ef4444; font-size: 13px;")
        
        QMessageBox.critical(
            self, 
            "Model Loading Error", 
            f"❌ Failed to load AI model:\n\n{error_message}"
        )
    
    def select_image(self):
        """Select an image file for prediction"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Plant Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            
            filename = os.path.basename(file_path)
            self.image_status_label.setText(f"🟢 {filename}")
            self.image_status_label.setStyleSheet("color: #10b981; font-weight: bold; font-size: 13px;")
            
            if self.model:
                self.predict_btn.setEnabled(True)
    
    def display_image(self, image_path):
        """Display the selected image with modern styling"""
        try:
            pixmap = QPixmap(image_path)
            
            # Scale image to fit display while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                400, 400, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setText("")
            
        except Exception as e:
            QMessageBox.critical(self, "Image Error", f"Failed to display image:\n{str(e)}")
    
    def predict_disease(self):
        """Run disease prediction on the selected image"""
        if not self.model or not self.current_image_path:
            QMessageBox.warning(
                self, 
                "Missing Requirements", 
                "Please load an AI model and select a plant image first."
            )
            return
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.predict_btn.setEnabled(False)
        self.results_text.clear()
        self.results_text.append("🔄 Analyzing plant health...\n")
        
        # Run prediction in separate thread
        self.prediction_worker = PredictionWorker(
            self.model, 
            self.current_image_path, 
            self.class_names,
            self.model_input_size,
            self.external_rescale
        )
        self.prediction_worker.finished.connect(self.on_prediction_finished)
        self.prediction_worker.error.connect(self.on_prediction_error)
        self.prediction_worker.start()
    
    def on_prediction_finished(self, result):
        """Handle prediction completion"""
        self.progress_bar.setVisible(False)
        self.predict_btn.setEnabled(True)
        
        # Format and display results
        self.display_results(result)
    
    def on_prediction_error(self, error_message):
        """Handle prediction error"""
        self.progress_bar.setVisible(False)
        self.predict_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Analysis Error", f"❌ Failed to analyze image:\n\n{error_message}")
        self.results_text.clear()
        self.results_text.append(f"❌ Analysis Error:\n{error_message}")
    
    def display_results(self, result):
        """Display prediction results in a modern format"""
        self.results_text.clear()
        
        # Header
        self.results_text.append("🌿 PLANT HEALTH ANALYSIS REPORT")
        self.results_text.append("=" * 50)
        self.results_text.append("")
        
        # Main prediction with emoji based on confidence
        confidence = result['percentage']
        if confidence > 80:
            status_emoji = "✅"
            status_color = "#10b981"
        elif confidence > 60:
            status_emoji = "⚠️"
            status_color = "#f59e0b"
        else:
            status_emoji = "❓"
            status_color = "#ef4444"
        
        self.results_text.append(f"{status_emoji} <span style='color: {status_color}; font-size: 16px; font-weight: bold;'>PREDICTION: {result['label']}</span>")
        self.results_text.append(f"📊 <span style='color: #60a5fa;'>CONFIDENCE: {confidence:.2f}%</span>")
        self.results_text.append("")
        
        # Top predictions
        self.results_text.append("🔍 TOP PREDICTIONS:")
        self.results_text.append("-" * 30)
        
        for pred in result.get('top_predictions', []):
            bar_width = int(pred['percentage'] / 100 * 20)
            bar = "█" * bar_width + "░" * (20 - bar_width)
            
            if pred['rank'] == 1:
                self.results_text.append(f"<span style='color: #10b981;'>🥇 {pred['class']}</span>")
                self.results_text.append(f"    {bar} {pred['percentage']:.2f}%")
            else:
                self.results_text.append(f"{pred['rank']}. {pred['class']}")
                self.results_text.append(f"    {bar} {pred['percentage']:.2f}%")
        
        self.results_text.append("")
        
        # Recommendations
        self.results_text.append("💡 RECOMMENDATIONS:")
        self.results_text.append("-" * 20)
        
        if "healthy" in result['label'].lower():
            self.results_text.append("• 🌱 Plant appears healthy")
            self.results_text.append("• 💧 Continue regular care")
            self.results_text.append("• 👀 Monitor for changes")
        else:
            self.results_text.append("• 🚨 Possible disease detected")
            self.results_text.append("• 🔍 Consult agricultural expert")
            self.results_text.append("• 📚 Research treatment options")
            self.results_text.append("• 🛡️ Consider preventive measures")
    
    def parse_class_name(self, class_name):
        """Retained for compatibility; not used in web-style results."""
        if "___" in class_name:
            parts = class_name.split("___")
            plant = parts[0].replace("_", " ").replace("(", " (").title()
            disease = parts[1].replace("_", " ").title()
        else:
            plant = class_name.replace("_", " ").title()
            disease = "Unknown"
        return plant, disease

    def load_class_names(self):
        """Load class names from 'leaf-map.json' if available, else use a fallback list."""
        mapping_path = os.path.join(os.getcwd(), "leaf-map.json")
        try:
            if os.path.exists(mapping_path):
                with open(mapping_path, "r", encoding="utf-8") as f:
                    idx_to_label = json.load(f)
                # Ensure correct order by numeric index
                ordered = [idx_to_label[str(i)] for i in range(len(idx_to_label))]
                return ordered
        except Exception:
            pass
        # Fallback to known PlantVillage order (matches many common models)
        return [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

    def get_model_input_size(self, model):
        """Infer input size (width, height) from the model input shape."""
        try:
            shape = model.input_shape
            # Typical shape: (None, H, W, C)
            if isinstance(shape, tuple) and len(shape) == 4:
                height, width = shape[1], shape[2]
                if isinstance(height, int) and isinstance(width, int):
                    return (width, height)
        except Exception:
            pass
        return (224, 224)

    def detect_external_rescale(self, model):
        """Return True if we should rescale externally (no Rescaling layer inside model)."""
        try:
            for layer in model.layers[:3]:
                if layer.__class__.__name__ == 'Rescaling':
                    return False
        except Exception:
            pass
        return True

def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("PlantGuard AI")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Plant Disease Detector")
    
    # Create and show the main window
    window = CropDiseaseDetectionApp()
    window.show()
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()