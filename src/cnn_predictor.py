import tensorflow as tf
import numpy as np
from PIL import Image
# import cv2  # Removed OpenCV import
import os
import h5py
from tensorflow import keras

class MediscopePredictor:
    def __init__(self, model_path="models/pneumonia_model.h5"):
        """
        Initialize the Mediscope AI predictor with the trained pneumonia detection CNN model.
        
        Args:
            model_path (str): Path to the trained CNN model file
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['Pneumonia', 'Normal']  # Class 0: Pneumonia, Class 1: Normal
        self.img_size = 150  # Model expects 150x150 images
        self.load_model()
        
    def load_model(self):
        """Load the trained pneumonia detection CNN model."""
        try:
            # Check if model file exists
            print(f"🔍 Checking for model file: {self.model_path}")
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                print("Please ensure the trained model file is available in the models folder.")
                print("For deployment, ensure the model file is included in your deployment package.")
                self.model = None
                return
            
            print(f"✅ Model file found: {self.model_path}")
            print(f"📁 File size: {os.path.getsize(self.model_path)} bytes")
            
            # Simple direct loading for TensorFlow 2.19
            print("🔄 Loading model with TensorFlow 2.19...")
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print("✅ Model loaded successfully")
            
            # Recompile with the correct settings to match training
            print("🔄 Compiling model...")
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
            print("✅ Pneumonia detection model loaded and compiled successfully")
            
        except Exception as e:
            print(f"❌ Error loading pneumonia detection model: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            print("Please ensure the model file is compatible and not corrupted.")
            print("For deployment issues, check if the model file is properly included.")
            self.model = None
    

    

    
    def preprocess_image(self, image_path):
        """
        Preprocess image to match the model's expected input format.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image ready for prediction
        """
        try:
            # Load image using PIL (grayscale)
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            
            if img is None:
                print(f"❌ Could not load image: {image_path}")
                return None
            
            # Resize to 150x150 (model's expected input size)
            img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Normalize pixel values to [0, 1]
            img_array = img_array.astype(np.float32) / 255.0
            
            # Add channel dimension to make it (150, 150, 1)
            img_array = np.expand_dims(img_array, axis=-1)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None
    
    def predict(self, image_path):
        """
        Predict pneumonia from chest X-ray image (binary classification).
        
        Args:
            image_path (str): Path to the chest X-ray image
            
        Returns:
            dict: Prediction results with confidence scores
        """
        print(f"🔍 Starting prediction for: {image_path}")
        
        # Try to load model if not already loaded
        if self.model is None:
            print("🔄 Model is None, attempting to load...")
            self.load_model()
        
        if self.model is None:
            print("❌ Model failed to load")
            return {
                "error": "Model not available. This feature requires the model file (pneumonia_model.h5) which may not be included in the deployment. Please ensure the model file is present in the models/ directory.",
                "prediction": None,
                "confidence": 0.0
            }
        
        print("✅ Model is loaded and ready")
        
        # Preprocess image
        processed_img = self.preprocess_image(image_path)
        if processed_img is None:
            return {
                "error": "Failed to preprocess image",
                "prediction": None,
                "confidence": 0.0
            }
        
        try:
            # Make prediction
            prediction = self.model.predict(processed_img, verbose=0)
            
            # Get the probability (sigmoid output)
            pneumonia_probability = float(prediction[0][0])
            
            # Determine predicted class (threshold 0.5)
            if pneumonia_probability > 0.5:
                predicted_class = "Pneumonia"
                confidence = pneumonia_probability
            else:
                predicted_class = "Normal"
                confidence = 1 - pneumonia_probability
            
            # Create results dictionary
            results = {
                "prediction": predicted_class,
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%",
                "class_scores": {
                    "pneumonia": pneumonia_probability * 100,
                    "normal": (1 - pneumonia_probability) * 100
                },
                "error": None
            }
            
            return results
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "prediction": None,
                "confidence": 0.0
            }
    
    def get_prediction_summary(self, result):
        """
        Get a summary of the prediction result.
        
        Args:
            result (dict): Prediction result from predict method
            
        Returns:
            str: Summary string
        """
        if result.get("error"):
            return f"Error: {result['error']}"
        
        prediction = result["prediction"]
        confidence = result["confidence_percentage"]
        
        if prediction == "Pneumonia":
            return f"⚠️ Pneumonia detected with {confidence} confidence. Please consult a healthcare professional."
        elif prediction == "Normal":
            return f"✅ Normal chest X-ray with {confidence} confidence."
        else:
            return f"❓ Uncertain result with {confidence} confidence."
