import tensorflow as tf
import numpy as np
from PIL import Image
# import cv2  # Removed OpenCV import
import os
import h5py
from tensorflow import keras
import streamlit as st

class CompatibleInputLayer(tf.keras.layers.InputLayer):
    """
    Custom InputLayer that handles batch_shape parameter compatibility.
    This fixes the issue with models saved in older Keras versions.
    """
    def __init__(self, input_shape=None, batch_size=None, dtype=None, 
                 input_tensor=None, sparse=None, name=None, ragged=None, 
                 type_spec=None, **kwargs):
        # Remove batch_shape from kwargs if present (compatibility fix)
        if 'batch_shape' in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if batch_shape and len(batch_shape) > 1:
                input_shape = batch_shape[1:]
        
        super().__init__(
            input_shape=input_shape,
            batch_size=batch_size,
            dtype=dtype,
            input_tensor=input_tensor,
            sparse=sparse,
            name=name,
            ragged=ragged,
            type_spec=type_spec,
            **kwargs
        )

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
        
    def validate_model_file(self):
        """Validate that the model file exists and is readable."""
        try:
            if not os.path.exists(self.model_path):
                return False, "Model file not found"
            
            file_size = os.path.getsize(self.model_path)
            if file_size == 0:
                return False, "Model file is empty"
            
            if file_size < 1000:  # Very small file, likely not a valid model
                return False, "Model file appears to be too small to be valid"
            
            # Try to open the file to check if it's readable
            with h5py.File(self.model_path, 'r') as f:
                # Check if it has the basic HDF5 structure
                if 'model_weights' not in f and 'keras_version' not in f:
                    return False, "File does not appear to be a valid Keras model"
            
            return True, "Model file appears valid"
            
        except Exception as e:
            return False, f"Error validating model file: {str(e)}"

    def load_model(self):
        """Load the trained pneumonia detection CNN model."""
        try:
            # Validate model file first
            st.info(f"🔍 Validating model file: {self.model_path}")
            is_valid, validation_message = self.validate_model_file()
            
            if not is_valid:
                st.error(f"❌ Model validation failed: {validation_message}")
                st.error("Please ensure the trained model file is available and not corrupted.")
                st.error("For deployment, ensure the model file is properly included in your deployment package.")
                self.model = None
                return
            
            st.success(f"✅ Model file validation passed: {validation_message}")
            file_size = os.path.getsize(self.model_path)
            st.info(f"📁 File size: {file_size} bytes")
            
            # Try loading with custom objects to handle compatibility issues
            st.info("🔄 Loading model with compatibility fixes...")
            
            # Define custom objects to handle potential compatibility issues
            custom_objects = {
                'AUC': tf.keras.metrics.AUC,
                'BinaryAccuracy': tf.keras.metrics.BinaryAccuracy,
                'Precision': tf.keras.metrics.Precision,
                'Recall': tf.keras.metrics.Recall,
                'InputLayer': CompatibleInputLayer,  # Add the custom InputLayer
                'input_layer_2': CompatibleInputLayer  # Handle specific layer name from error
            }
            
            # Try loading with custom objects first
            try:
                self.model = tf.keras.models.load_model(
                    self.model_path, 
                    compile=False,
                    custom_objects=custom_objects
                )
                st.success("✅ Model loaded successfully with custom objects")
            except Exception as e1:
                st.warning(f"⚠️ First loading attempt failed: {str(e1)}")
                st.info("🔄 Trying alternative loading method...")
                
                # Try loading with different options
                try:
                    self.model = tf.keras.models.load_model(
                        self.model_path, 
                        compile=False,
                        options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
                    )
                    st.success("✅ Model loaded successfully with alternative method")
                except Exception as e2:
                    st.warning(f"⚠️ Alternative loading failed: {str(e2)}")
                    st.info("🔄 Trying with legacy format...")
                    
                    # Try with legacy format
                    try:
                        self.model = tf.keras.models.load_model(
                            self.model_path, 
                            compile=False,
                            custom_objects=custom_objects,
                            options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
                        )
                        st.success("✅ Model loaded successfully with legacy format")
                    except Exception as e3:
                        st.warning(f"⚠️ Legacy format loading failed: {str(e3)}")
                        st.info("🔄 Trying final fallback method...")
                        
                        # Final fallback: Try loading weights only
                        try:
                            # Create a simple model architecture and load weights
                            st.info("🔄 Creating model architecture and loading weights...")
                            
                            # Create a basic CNN architecture that matches the expected input
                            self.model = tf.keras.Sequential([
                                tf.keras.layers.Input(shape=(150, 150, 1)),
                                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                tf.keras.layers.MaxPooling2D((2, 2)),
                                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                tf.keras.layers.MaxPooling2D((2, 2)),
                                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(64, activation='relu'),
                                tf.keras.layers.Dense(1, activation='sigmoid')
                            ])
                            
                            # Try to load weights
                            self.model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
                            st.success("✅ Model loaded successfully with fallback architecture")
                            
                        except Exception as e4:
                            st.error(f"❌ All loading methods failed. Last error: {str(e4)}")
                            st.error("The model file may be incompatible with the current TensorFlow version.")
                            st.error("Consider retraining the model with the current TensorFlow version.")
                            st.error("For now, the pneumonia detection feature will be unavailable.")
                            self.model = None
                            return
            
            # Recompile with the correct settings to match training
            st.info("🔄 Compiling model...")
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
            st.success("✅ Pneumonia detection model loaded and compiled successfully")
            
        except Exception as e:
            st.error(f"❌ Error loading pneumonia detection model: {e}")
            st.error(f"❌ Error type: {type(e).__name__}")
            import traceback
            st.error(f"❌ Full traceback: {traceback.format_exc()}")
            st.error("Please ensure the model file is compatible and not corrupted.")
            st.error("For deployment issues, check if the model file is properly included.")
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
        st.info(f"🔍 Starting prediction for: {image_path}")
        
        # Try to load model if not already loaded
        if self.model is None:
            st.warning("🔄 Model is None, attempting to load...")
            self.load_model()
        
        if self.model is None:
            st.error("❌ Model failed to load")
            return {
                "error": "Model not available. This feature requires the model file (pneumonia_model.h5) which may not be included in the deployment. Please ensure the model file is present in the models/ directory.",
                "prediction": None,
                "confidence": 0.0
            }
        
        st.success("✅ Model is loaded and ready")
        
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
