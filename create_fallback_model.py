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
    
    # Disable TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("🔄 Creating fallback CNN model...")
    
    # Create a simple CNN model with standard layers
    model = keras.Sequential([
        # Input layer - using standard Input instead of InputLayer
        keras.layers.Input(shape=(150, 150, 1), name='input_1'),
        
        # First convolutional block
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
        keras.layers.MaxPooling2D((2, 2), name='pool1'),
        keras.layers.BatchNormalization(name='bn1'),
        
        # Second convolutional block
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        keras.layers.MaxPooling2D((2, 2), name='pool2'),
        keras.layers.BatchNormalization(name='bn2'),
        
        # Third convolutional block
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        keras.layers.MaxPooling2D((2, 2), name='pool3'),
        keras.layers.BatchNormalization(name='bn3'),
        
        # Fourth convolutional block
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4'),
        keras.layers.MaxPooling2D((2, 2), name='pool4'),
        keras.layers.BatchNormalization(name='bn4'),
        
        # Flatten and dense layers
        keras.layers.Flatten(name='flatten'),
        keras.layers.Dense(512, activation='relu', name='dense1'),
        keras.layers.Dropout(0.5, name='dropout1'),
        keras.layers.Dense(256, activation='relu', name='dense2'),
        keras.layers.Dropout(0.3, name='dropout2'),
        keras.layers.Dense(1, activation='sigmoid', name='output')  # Binary classification
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
    model.save(model_path, save_format='h5')
    
    print(f"✅ Fallback model created and saved to: {model_path}")
    print("📊 Model Summary:")
    model.summary()
    
    # Test the model
    print("\n🧪 Testing model with dummy input...")
    dummy_input = np.random.random((1, 150, 150, 1))
    prediction = model.predict(dummy_input, verbose=0)
    print(f"✅ Test prediction shape: {prediction.shape}")
    print(f"✅ Test prediction value: {prediction[0][0]:.4f}")
    
    # Also save as the main model file
    main_model_path = 'models/pneumonia_model.h5'
    model.save(main_model_path, save_format='h5')
    print(f"✅ Model also saved as main model: {main_model_path}")
    
    return model_path

if __name__ == "__main__":
    create_fallback_model()
