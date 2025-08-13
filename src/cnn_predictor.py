import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import h5py

class PneumoniaPredictor:
    def __init__(self, model_path="models/improved_pneumonia_cnn.h5"):
        """
        Initialize the pneumonia predictor with the trained CNN model.
        
        Args:
            model_path (str): Path to the trained CNN model file
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained CNN model."""
        try:
            # Try loading with custom_objects to handle compatibility issues
            custom_objects = {}
            
            # First attempt: Try loading with compile=False and custom DTypePolicy
            try:
                # Define a custom DTypePolicy for compatibility
                class CompatibleDTypePolicy:
                    def __init__(self, name='float32'):
                        self.name = name
                
                custom_objects['DTypePolicy'] = CompatibleDTypePolicy
                
                self.model = tf.keras.models.load_model(
                    self.model_path, 
                    compile=False,
                    custom_objects=custom_objects
                )
                print("✅ CNN model loaded successfully (with custom DTypePolicy)")
                return
            except Exception as e1:
                print(f"⚠️  First attempt failed: {e1}")
            
            # Second attempt: Try loading with custom InputLayer and DTypePolicy
            try:
                # Define a custom InputLayer that ignores batch_shape
                class CompatibleInputLayer(tf.keras.layers.InputLayer):
                    def __init__(self, **kwargs):
                        # Remove batch_shape if present
                        if 'batch_shape' in kwargs:
                            del kwargs['batch_shape']
                        super().__init__(**kwargs)
                
                # Define a custom DTypePolicy for compatibility
                class CompatibleDTypePolicy:
                    def __init__(self, name='float32'):
                        self.name = name
                
                custom_objects['InputLayer'] = CompatibleInputLayer
                custom_objects['DTypePolicy'] = CompatibleDTypePolicy
                
                self.model = tf.keras.models.load_model(
                    self.model_path, 
                    compile=False,
                    custom_objects=custom_objects
                )
                print("✅ CNN model loaded successfully (with custom InputLayer and DTypePolicy)")
                return
            except Exception as e2:
                print(f"⚠️  Second attempt failed: {e2}")
            
            # Third attempt: Try loading with safe_mode
            try:
                self.model = tf.keras.models.load_model(
                    self.model_path, 
                    compile=False,
                    safe_mode=True
                )
                print("✅ CNN model loaded successfully (safe_mode)")
                return
            except Exception as e3:
                print(f"⚠️  Third attempt failed: {e3}")
            
            # Fourth attempt: Try loading weights only and reconstructing model
            try:
                # Load just the weights
                with h5py.File(self.model_path, 'r') as f:
                    # Check if model_weights exists
                    if 'model_weights' in f:
                        print("📦 Attempting to load weights and reconstruct model...")
                        
                        # Create a simple model architecture based on the weights
                        self.model = self.create_compatible_model()
                        
                        # Load weights
                        self.model.load_weights(self.model_path)
                        print("✅ CNN model loaded successfully (weights reconstruction)")
                        return
            except Exception as e4:
                print(f"⚠️  Fourth attempt failed: {e4}")
            
            # If all attempts fail, provide detailed error
            raise Exception(f"All loading attempts failed. Last error: {e4 if 'e4' in locals() else e3 if 'e3' in locals() else e2 if 'e2' in locals() else e1}")
            
        except Exception as e:
            print(f"❌ Error loading CNN model: {e}")
            print("💡 This might be due to TensorFlow version compatibility.")
            print("   Try updating TensorFlow or using a compatible version.")
            self.model = None
    
    def create_compatible_model(self):
        """Create a compatible model architecture based on the saved weights."""
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(224, 224, 3)),
            
            # First conv block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Second conv block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Third conv block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Fourth conv block
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Flatten and dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def preprocess_image(self, image_path):
        """
        Preprocess the input image for CNN prediction.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            # Load and resize image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not load image")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size (assuming 224x224, adjust if different)
            img = cv2.resize(img, (224, 224))
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None
    
    def predict(self, image_path):
        """
        Predict pneumonia from chest X-ray image.
        
        Args:
            image_path (str): Path to the chest X-ray image
            
        Returns:
            dict: Prediction results with confidence scores
        """
        if self.model is None:
            return {
                "error": "Model not loaded. Please check TensorFlow compatibility.",
                "prediction": None,
                "confidence": 0.0
            }
        
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
            
            # Assuming binary classification: 0 = Normal, 1 = Pneumonia
            # Adjust based on your model's output format
            confidence = float(prediction[0][0]) if prediction.shape[1] == 1 else float(np.max(prediction))
            
            # Determine prediction class
            if prediction.shape[1] == 1:
                # Binary classification
                predicted_class = "Pneumonia" if confidence > 0.5 else "Normal"
                confidence = confidence if predicted_class == "Pneumonia" else 1 - confidence
            else:
                # Multi-class classification
                predicted_class = "Pneumonia" if np.argmax(prediction) == 1 else "Normal"
                confidence = float(np.max(prediction))
            
            return {
                "prediction": predicted_class,
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%",
                "error": None
            }
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "prediction": None,
                "confidence": 0.0
            }
    
    def get_prediction_summary(self, result):
        """
        Generate a human-readable summary of the prediction results.
        
        Args:
            result (dict): Prediction result from predict() method
            
        Returns:
            str: Human-readable summary
        """
        if result.get("error"):
            return f"❌ Error: {result['error']}"
        
        prediction = result["prediction"]
        confidence = result["confidence_percentage"]
        
        if prediction == "Pneumonia":
            return f"🔴 **Pneumonia Detected**\n\nConfidence: {confidence}\n\n⚠️ **Important**: This is an AI-assisted analysis. Please consult with a healthcare professional for proper diagnosis and treatment."
        else:
            return f"🟢 **Normal Chest X-ray**\n\nConfidence: {confidence}\n\n✅ The image appears to show a normal chest X-ray without signs of pneumonia."
