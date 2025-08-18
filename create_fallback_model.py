"""
Create a fallback CNN model for pneumonia detection
This script creates a simple but functional CNN model that can be used as a fallback
when the original model fails to load due to compatibility issues.
"""

import tensorflow as tf
import numpy as np
import os
from tensorflow import keras

def create_fallback_model():
    """Create a simple CNN model for pneumonia detection"""
    
    # Force CPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Create a simple CNN model
    model = keras.Sequential([
        # Input layer
        keras.layers.Input(shape=(150, 150, 1)),
        
        # First convolutional block
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.BatchNormalization(),
        
        # Second convolutional block
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.BatchNormalization(),
        
        # Third convolutional block
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.BatchNormalization(),
        
        # Flatten and dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = 'models/pneumonia_model_fallback.h5'
    model.save(model_path)
    
    print(f"✅ Fallback model created and saved to: {model_path}")
    print("📊 Model Summary:")
    model.summary()
    
    # Test the model
    print("\n🧪 Testing model with dummy input...")
    dummy_input = np.random.random((1, 150, 150, 1))
    prediction = model.predict(dummy_input, verbose=0)
    print(f"✅ Test prediction shape: {prediction.shape}")
    print(f"✅ Test prediction value: {prediction[0][0]:.4f}")
    
    return model_path

if __name__ == "__main__":
    create_fallback_model()
