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
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                print("Please ensure the trained model file is available in the models folder.")
                print("For deployment, ensure the model file is included in your deployment package.")
                self.model = None
                return
            
            # Try different loading strategies for compatibility
            self.model = self._load_model_with_fallback()
            
            if self.model is not None:
                # Recompile with the correct settings to match training
                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
                )
                print("✅ Pneumonia detection model loaded successfully")
            else:
                print("❌ Failed to load model with all fallback strategies")
            
        except Exception as e:
            print(f"❌ Error loading pneumonia detection model: {e}")
            print("Please ensure the model file is compatible and not corrupted.")
            print("For deployment issues, check if the model file is properly included.")
            self.model = None
    
    def _load_model_with_fallback(self):
        """Load model with multiple fallback strategies for compatibility"""
        strategies = [
            self._load_model_strategy_1,  # Standard load
            self._load_model_strategy_2,  # With custom objects
            self._load_model_strategy_3,  # With custom objects and compile=False
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                print(f"🔄 Trying loading strategy {i}...")
                model = strategy()
                if model is not None:
                    print(f"✅ Strategy {i} successful")
                    return model
            except Exception as e:
                print(f"❌ Strategy {i} failed: {str(e)[:100]}...")
                continue
        
        return None
    
    def _load_model_strategy_1(self):
        """Standard model loading"""
        return tf.keras.models.load_model(self.model_path, compile=False)
    
    def _load_model_strategy_2(self):
        """Load with custom objects to handle version compatibility"""
        # Define custom objects to handle version differences
        custom_objects = {
            'InputLayer': tf.keras.layers.InputLayer,
        }
        
        # Try to load with custom objects
        return tf.keras.models.load_model(
            self.model_path, 
            compile=False, 
            custom_objects=custom_objects
        )
    
    def _load_model_strategy_3(self):
        """Load with more comprehensive custom objects"""
        # More comprehensive custom objects for older TensorFlow versions
        custom_objects = {
            'InputLayer': tf.keras.layers.InputLayer,
            'Adam': tf.keras.optimizers.Adam,
            'binary_crossentropy': tf.keras.losses.binary_crossentropy,
            'accuracy': tf.keras.metrics.binary_accuracy,
        }
        
        # Try to load with custom objects and handle batch_shape issue
        try:
            return tf.keras.models.load_model(
                self.model_path, 
                compile=False, 
                custom_objects=custom_objects
            )
        except Exception as e:
            if 'batch_shape' in str(e):
                # Handle the batch_shape compatibility issue
                print("🔄 Handling batch_shape compatibility issue...")
                return self._load_model_with_batch_shape_fix()
            raise e
    
    def _load_model_with_batch_shape_fix(self):
        """Fix batch_shape compatibility issue by loading with custom InputLayer"""
        import h5py
        
        # Create a custom InputLayer that ignores batch_shape
        class CompatibleInputLayer(tf.keras.layers.InputLayer):
            def __init__(self, input_shape=None, **kwargs):
                # Remove batch_shape from kwargs if present
                kwargs.pop('batch_shape', None)
                super().__init__(input_shape=input_shape, **kwargs)
        
        custom_objects = {
            'InputLayer': CompatibleInputLayer,
            'Adam': tf.keras.optimizers.Adam,
            'binary_crossentropy': tf.keras.losses.binary_crossentropy,
            'accuracy': tf.keras.metrics.binary_accuracy,
        }
        
        return tf.keras.models.load_model(
            self.model_path, 
            compile=False, 
            custom_objects=custom_objects
        )
    

    
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
        # Try to load model if not already loaded
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            return {
                "error": "Model not available. This feature requires the model file (pneumonia_model.h5) which may not be included in the deployment. Please ensure the model file is present in the models/ directory.",
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
