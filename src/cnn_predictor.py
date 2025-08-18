import os
import numpy as np
from PIL import Image
# import cv2  # Removed OpenCV import
import h5py

# Configure environment before importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow imported successfully")
except Exception as e:
    print(f"⚠️ TensorFlow import failed: {e}")
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None

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
        self.tensorflow_available = TENSORFLOW_AVAILABLE
        
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow is not available. Model loading will be skipped.")
            print("💡 The app will run in limited mode - only chatbot functionality will be available.")
            return
            
        # Force CPU usage for deployment compatibility
        self._configure_tensorflow()
        self.load_model()
        
    def _configure_tensorflow(self):
        """Configure TensorFlow for deployment compatibility."""
        try:
            # Force CPU usage to avoid CUDA issues
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # Configure TensorFlow to use CPU only
            tf.config.set_visible_devices([], 'GPU')
            
            # Set memory growth to avoid memory issues
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"GPU memory growth setting failed: {e}")
                    
            print("✅ TensorFlow configured for CPU-only deployment")
            
        except Exception as e:
            print(f"⚠️ TensorFlow configuration warning: {e}")
        
    def load_model(self):
        """Load the trained pneumonia detection CNN model."""
        if not TENSORFLOW_AVAILABLE:
            print("❌ Cannot load model: TensorFlow is not available")
            return
            
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                print("Please ensure the trained model file is available in the models folder.")
                self.model = None
                return
            
            # Load the model with CPU-only configuration
            print("🔄 Loading pneumonia detection model...")
            
            # Try different loading strategies
            loading_strategies = [
                # Strategy 1: Load with custom objects and ignore errors
                lambda: tf.keras.models.load_model(
                    self.model_path, 
                    compile=False,
                    options=tf.saved_model.LoadOptions(experimental_io_device='/cpu:0')
                ),
                # Strategy 2: Load with custom objects
                lambda: tf.keras.models.load_model(
                    self.model_path, 
                    compile=False,
                    custom_objects={}
                ),
                # Strategy 3: Load with custom objects and handle InputLayer
                lambda: tf.keras.models.load_model(
                    self.model_path, 
                    compile=False,
                    custom_objects={
                        'InputLayer': tf.keras.layers.InputLayer
                    }
                ),
                # Strategy 4: Load with custom objects and handle all potential issues
                lambda: tf.keras.models.load_model(
                    self.model_path, 
                    compile=False,
                    custom_objects={
                        'InputLayer': tf.keras.layers.InputLayer,
                        'Dense': tf.keras.layers.Dense,
                        'Conv2D': tf.keras.layers.Conv2D,
                        'MaxPooling2D': tf.keras.layers.MaxPooling2D,
                        'Flatten': tf.keras.layers.Flatten,
                        'Dropout': tf.keras.layers.Dropout,
                        'BatchNormalization': tf.keras.layers.BatchNormalization
                    }
                )
            ]
            
            for i, strategy in enumerate(loading_strategies):
                try:
                    print(f"🔄 Trying loading strategy {i+1}...")
                    self.model = strategy()
                    
                    # Test the model with a dummy input
                    dummy_input = np.random.random((1, self.img_size, self.img_size, 1))
                    test_prediction = self.model.predict(dummy_input, verbose=0)
                    
                    if test_prediction is not None and test_prediction.shape[1] == 1:
                        print(f"✅ Pneumonia detection model loaded successfully with strategy {i+1}")
                        break
                    else:
                        raise Exception("Model test prediction failed")
                        
                except Exception as e:
                    print(f"⚠️ Strategy {i+1} failed: {str(e)}")
                    if i == len(loading_strategies) - 1:  # Last strategy
                        raise e
                    continue
            
            # Recompile with the correct settings to match training
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
            
            print("✅ Pneumonia detection model loaded and tested successfully")
            
        except Exception as e:
            print(f"❌ Error loading pneumonia detection model: {e}")
            print("Attempting to load fallback model...")
            
            # Try to load fallback model
            fallback_path = "models/pneumonia_model_fallback.h5"
            if os.path.exists(fallback_path):
                try:
                    print("🔄 Loading fallback model...")
                    self.model = tf.keras.models.load_model(fallback_path, compile=False)
                    self.model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss='binary_crossentropy',
                        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
                    )
                    print("✅ Fallback model loaded successfully")
                except Exception as fallback_error:
                    print(f"❌ Fallback model also failed: {fallback_error}")
                    self.model = None
            else:
                print("❌ No fallback model available. Please run create_fallback_model.py")
                self.model = None
    
    def is_model_available(self):
        """
        Check if the model is available for predictions.
        
        Returns:
            bool: True if model is available, False otherwise
        """
        return self.model is not None and TENSORFLOW_AVAILABLE

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
        # Check if TensorFlow is available
        if not TENSORFLOW_AVAILABLE:
            return {
                "error": "TensorFlow is not available. Model prediction is not supported in this environment.",
                "prediction": None,
                "confidence": 0.0
            }
        
        # Try to load model if not already loaded
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            return {
                "error": "Model not available. This feature requires the model file which is not included in the deployment.",
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
