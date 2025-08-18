#!/usr/bin/env python3
"""
Model conversion script to fix compatibility issues with TensorFlow versions.
This script loads the old model and saves it in a format compatible with newer TensorFlow versions.
"""

import tensorflow as tf
import os
import sys

def convert_model():
    """Convert the pneumonia model to a compatible format."""
    
    model_path = "models/pneumonia_model.h5"
    output_path = "models/pneumonia_model_converted.h5"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        print("🔄 Loading model with custom_objects to handle compatibility...")
        
        # Try to load with custom objects to handle the batch_shape issue
        custom_objects = {}
        
        # Load the model with custom objects
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False
        )
        
        print("✅ Model loaded successfully!")
        
        # Recompile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        print("🔄 Saving converted model...")
        
        # Save the model in a compatible format
        model.save(output_path, save_format='h5')
        
        print(f"✅ Model converted and saved to: {output_path}")
        print("🔄 You can now update your model path in the code to use the converted model.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        
        # Try alternative approach with different TensorFlow version
        print("🔄 Trying alternative conversion approach...")
        try:
            # Load model weights only
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
            
            # Create a new model with the same architecture
            input_shape = (150, 150, 1)
            new_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            # Copy weights
            for old_layer, new_layer in zip(model.layers, new_model.layers):
                if hasattr(old_layer, 'get_weights'):
                    new_layer.set_weights(old_layer.get_weights())
            
            # Compile the new model
            new_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
            
            # Save the new model
            new_model.save(output_path, save_format='h5')
            print(f"✅ Model converted using alternative approach: {output_path}")
            return True
            
        except Exception as e2:
            print(f"❌ Alternative approach also failed: {e2}")
            return False

if __name__ == "__main__":
    print("🔧 Pneumonia Model Converter")
    print("=" * 40)
    
    success = convert_model()
    
    if success:
        print("\n✅ Conversion completed successfully!")
        print("📝 Next steps:")
        print("1. Update your model path in cnn_predictor.py to use 'models/pneumonia_model_converted.h5'")
        print("2. Restart your Streamlit application")
    else:
        print("\n❌ Conversion failed!")
        print("💡 Try using TensorFlow 2.15.0 or earlier versions for compatibility.")
