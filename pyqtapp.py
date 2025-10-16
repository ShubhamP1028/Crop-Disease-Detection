import sys
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QTextEdit, QWidget,
                             QProgressBar, QMessageBox, QFrame, QGridLayout, QScrollArea, QSizePolicy,
                             QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PyQt6.QtGui import QPixmap, QFont, QPalette, QColor, QIcon, QLinearGradient, QPainter, QBrush, QPen


class CustomButton(QPushButton):
    def __init__(self, text, primary=False):
        super().__init__(text)
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.primary = primary
        self._setup_style()
        
    def _setup_style(self):
        if self.primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #4a7c59;
                    color: white;
                    border: 2px solid #3a6b49;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 14px;
                    padding: 10px 20px;
                }
                QPushButton:hover {
                    background-color: #5a8c69;
                    border-color: #4a7c59;
                }
                QPushButton:pressed {
                    background-color: #3a6b49;
                }
                QPushButton:disabled {
                    background-color: #6a6a6a;
                    color: #9a9a9a;
                    border-color: #5a5a5a;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #6a6a6a;
                    color: white;
                    border: 2px solid #5a5a5a;
                    border-radius: 8px;
                    font-weight: normal;
                    font-size: 13px;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #7a7a7a;
                    border-color: #6a6a6a;
                }
                QPushButton:pressed {
                    background-color: #5a5a5a;
                }
                QPushButton:disabled {
                    background-color: #4a4a4a;
                    color: #8a8a8a;
                    border-color: #3a3a3a;
                }
            """)


class SimpleHeader(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(60)
        self._setup_style()

    def _setup_style(self):
        self.setStyleSheet("""
            QLabel {
                background-color: #2d5a2d;
                color: white;
                font-size: 24px;
                font-weight: bold;
                border: 2px solid #1a4a1a;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
            }
        """)


class SimpleCard(QFrame):
    def __init__(self, title=None):
        super().__init__()
        self.setObjectName("card")
        self.setStyleSheet("""
            #card {
                background-color: #f8f8f8;
                border: 2px solid #d0d0d0;
                border-radius: 8px;
                padding: 0;
            }
        """)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)
        
        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet("""
                QLabel {
                    color: #2d5a2d;
                    font-size: 16px;
                    font-weight: bold;
                    padding-bottom: 8px;
                    border-bottom: 1px solid #c0c0c0;
                    margin-bottom: 8px;
                }
            """)
            self.layout.addWidget(title_label)


class SimpleProgressBar(QProgressBar):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(20)
        self.setTextVisible(True)
        self.setStyleSheet("""
            QProgressBar {
                background-color: #e0e0e0;
                border: 1px solid #c0c0c0;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4a7c59;
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
        """Initialize the user interface"""
        self.setWindowTitle("Crop Disease Detection Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Left panel for controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel for image display and results
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
    
    def create_left_panel(self):
        """Create left control panel"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)
        
        # App header
        header = SimpleHeader("Crop Disease Detection")
        left_layout.addWidget(header)
        
        # Model section
        model_card = SimpleCard("Model Management")
        model_layout = QVBoxLayout()
        
        self.load_model_btn = CustomButton("Load Model (.h5)", primary=True)
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)
        
        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setStyleSheet("color: #666; font-size: 12px; font-style: italic;")
        model_layout.addWidget(self.model_status_label)
        
        model_card.layout.addLayout(model_layout)
        left_layout.addWidget(model_card)
        
        # Image section
        image_card = SimpleCard("Image Selection")
        image_layout = QVBoxLayout()
        
        self.select_image_btn = CustomButton("Select Plant Image", primary=False)
        self.select_image_btn.clicked.connect(self.select_image)
        self.select_image_btn.setEnabled(False)
        image_layout.addWidget(self.select_image_btn)
        
        self.image_status_label = QLabel("No image selected")
        self.image_status_label.setStyleSheet("color: #666; font-size: 12px; font-style: italic;")
        image_layout.addWidget(self.image_status_label)
        
        image_card.layout.addLayout(image_layout)
        left_layout.addWidget(image_card)
        
        # Analysis section
        analysis_card = SimpleCard("Analysis")
        analysis_layout = QVBoxLayout()
        
        self.predict_btn = CustomButton("Analyze Plant Health", primary=True)
        self.predict_btn.clicked.connect(self.predict_disease)
        self.predict_btn.setEnabled(False)
        analysis_layout.addWidget(self.predict_btn)
        
        self.progress_bar = SimpleProgressBar()
        self.progress_bar.setVisible(False)
        analysis_layout.addWidget(self.progress_bar)
        
        analysis_card.layout.addLayout(analysis_layout)
        left_layout.addWidget(analysis_card)
        
        # Instructions
        instructions_card = SimpleCard("Instructions")
        instructions_layout = QVBoxLayout()
        
        instructions_text = QLabel("""
        1. Load your trained .h5 model file
        2. Select a plant leaf image
        3. Click 'Analyze Plant Health'
        4. View results and recommendations
        
        Supported formats: JPG, PNG, JPEG
        Recommended: Clear leaf images
        """)
        instructions_text.setStyleSheet("color: #555; font-size: 12px; line-height: 1.4;")
        instructions_text.setWordWrap(True)
        instructions_layout.addWidget(instructions_text)
        
        instructions_card.layout.addLayout(instructions_layout)
        left_layout.addWidget(instructions_card)
        
        left_layout.addStretch()
        return left_widget
    
    def create_right_panel(self):
        """Create right display panel"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)
        
        # Image display section
        image_card = SimpleCard("Plant Image")
        image_layout = QVBoxLayout()
        
        # Scrollable image area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(300)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 2px dashed #ccc;")
        self.image_label.setText("No image loaded\n\nSelect an image to begin analysis")
        self.image_label.setMinimumSize(400, 300)
        
        scroll_area.setWidget(self.image_label)
        image_layout.addWidget(scroll_area)
        
        image_card.layout.addLayout(image_layout)
        right_layout.addWidget(image_card)
        
        # Results section
        results_card = SimpleCard("Analysis Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        self.results_text.setPlaceholderText("Analysis results will appear here after prediction...")
        self.results_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
                background-color: #fafafa;
                color: #333;
            }
        """)
        results_layout.addWidget(self.results_text)
        
        results_card.layout.addLayout(results_layout)
        right_layout.addWidget(results_card)
        
        return right_widget
    
    def apply_styles(self):
        """Apply clean, simple styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
    
    def load_model(self):
        """Load the trained model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
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
            self.model_status_label.setText("Model loaded successfully")
            self.model_status_label.setStyleSheet("color: #2d5a2d; font-weight: bold; font-size: 12px;")
            self.select_image_btn.setEnabled(True)
            # Infer model input size and preprocessing behavior
            self.model_input_size = self.get_model_input_size(self.model)
            self.external_rescale = self.detect_external_rescale(self.model)
            
            # Update UI
            QMessageBox.information(self, "Success", "Model loaded successfully!")
        else:
            self.model_status_label.setText("Failed to load model")
            self.model_status_label.setStyleSheet("color: #a55a5a; font-size: 12px;")
    
    def on_model_error(self, error_message):
        """Handle model loading error"""
        self.progress_bar.setVisible(False)
        self.model_status_label.setText("Error loading model")
        self.model_status_label.setStyleSheet("color: #a55a5a; font-size: 12px;")
        
        QMessageBox.critical(self, "Error", f"Error loading model:\n\n{error_message}")
    
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
            self.image_status_label.setText(f"Image loaded: {filename}")
            self.image_status_label.setStyleSheet("color: #2d5a2d; font-weight: bold; font-size: 12px;")
            
            if self.model:
                self.predict_btn.setEnabled(True)
    
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            pixmap = QPixmap(image_path)
            
            # Scale image to fit display while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                400, 300, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setText("")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display image:\n{str(e)}")
    
    def predict_disease(self):
        """Run disease prediction on the selected image"""
        if not self.model or not self.current_image_path:
            QMessageBox.warning(
                self, 
                "Warning", 
                "Please load a model and select an image first."
            )
            return
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.predict_btn.setEnabled(False)
        self.results_text.clear()
        self.results_text.append("Analyzing plant health...")
        
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
        
        QMessageBox.critical(self, "Prediction Error", f"Error during prediction:\n\n{error_message}")
        self.results_text.clear()
        self.results_text.append(f"Error: {error_message}")
    
    def display_results(self, result):
        """Display prediction results in a simple format"""
        self.results_text.clear()
        
        # Main title
        self.results_text.append("Analysis Results")
        self.results_text.append("=" * 50)
        
        # Predicted label and confidence
        self.results_text.append(f"Prediction: {result['label']}")
        self.results_text.append(f"Confidence: {result['percentage']:.2f}%")
        self.results_text.append("")
        
        # Top predictions (up to 10)
        self.results_text.append("Top Predictions:")
        self.results_text.append("-" * 30)
        for pred in result.get('top_predictions', [])[:10]:
            self.results_text.append(f"{pred['rank']}. {pred['class']} - {pred['percentage']:.2f}%")
        
        self.results_text.append("")
        
        # Simple recommendations
        label = result['label']
        if "healthy" in label.lower():
            self.results_text.append("Recommendations:")
            self.results_text.append("- Your plant appears healthy")
            self.results_text.append("- Continue current care routine")
            self.results_text.append("- Monitor regularly for changes")
        else:
            self.results_text.append("Recommendations:")
            self.results_text.append("- Disease detected - consult agricultural expert")
            self.results_text.append("- Consider appropriate treatment")
            self.results_text.append("- Monitor other plants for spread")
    
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
    app.setApplicationName("Crop Disease Detection")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("AI Plant Doctor")
    
    # Create and show the main window
    window = CropDiseaseDetectionApp()
    window.show()
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
