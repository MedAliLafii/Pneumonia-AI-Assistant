import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os

# Page configuration
st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="",
    layout="centered"
)

# Custom CSS for clean, professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .stFileUploader {
        border: 2px dashed #667eea !important;
        border-radius: 15px !important;
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
        padding: 2rem !important;
        text-align: center !important;
        color: white !important;
        transition: all 0.3s ease !important;
        margin: 2rem 0 !important;
    }
    
    .stFileUploader:hover {
        border-color: #764ba2 !important;
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%) !important;
    }
    
    .stFileUploader > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .stFileUploader label {
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    
    .result-box {
        background: linear-gradient(135deg, #2d5a2d 0%, #1e3a1e 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        margin: 2rem 0;
        text-align: center;
        color: white;
    }
    
    .result-box.pneumonia {
        background: linear-gradient(135deg, #5a2d2d 0%, #3a1e1e 100%);
        border-left-color: #dc3545;
        color: white;
    }
    
    .confidence-bar {
        background: #2c3e50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        height: 20px;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .confidence-fill.high {
        background: linear-gradient(90deg, #dc3545 0%, #fd7e14 100%);
    }
    

    
    .footer {
        text-align: center;
        color: #6c757d;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'page_loaded' not in st.session_state:
    st.session_state.page_loaded = False

# Loading screen
if not st.session_state.page_loaded:
    st.markdown("""
    <style>
    .loading-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        color: white;
        font-family: 'Inter', sans-serif;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-top: 4px solid white;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 20px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        font-size: 1.2rem;
        font-weight: 500;
        text-align: center;
    }
    
    .loading-subtitle {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-top: 10px;
    }
    </style>
    
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">Loading AI Pneumonia Detection</div>
        <div class="loading-subtitle">Initializing deep learning model...</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Simulate loading time and mark as loaded
    import time
    time.sleep(2)
    st.session_state.page_loaded = True
    st.rerun()

@st.cache_resource
def load_model():
    """Load the trained CNN model"""
    try:
        model_path = "improved_pneumonia_cnn.h5"
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            return model
        else:
            st.error(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize to model input size (224x224)
    image = image.resize((224, 224))
    
    # Convert to RGB if grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_pneumonia(image):
    """Make prediction using the loaded model"""
    if st.session_state.model is None:
        return None, 0.0
    
    try:
        # Preprocess image
        processed_img = preprocess_image(image)
        
        # Make prediction
        prediction = st.session_state.model.predict(processed_img, verbose=0)
        confidence = float(prediction[0][0])
        
        # Determine class
        if confidence > 0.5:
            result = "PNEUMONIA"
        else:
            result = "NORMAL"
            confidence = 1 - confidence
        
        return result, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, 0.0

# Sidebar with project information
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
        <h3> AI Pneumonia Detection</h3>
        <p style="font-size: 0.9rem; margin-bottom: 1rem;">Advanced deep learning system for chest X-ray analysis</p>
        <h4>📊 Model Performance</h4>
        <ul style="font-size: 0.85rem; margin-bottom: 1rem;">
            <li>Accuracy: 85.1%</li>
            <li>Precision: 95.4%</li>
            <li>Recall: 80.0%</li>
        </ul>
        <h4>🛠️ Technology</h4>
        <ul style="font-size: 0.85rem; margin-bottom: 1rem;">
            <li>TensorFlow/Keras</li>
            <li>Convolutional Neural Networks</li>
            <li>Streamlit Web App</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header"> AI Pneumonia Detection</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6c757d; font-size: 1.1rem;">Upload a chest X-ray image to get instant AI-powered analysis</p>', unsafe_allow_html=True)

# Load model
if not st.session_state.model_loaded:
    with st.spinner("Loading AI model..."):
        st.session_state.model = load_model()
        st.session_state.model_loaded = True

# File upload with better styling
uploaded_file = st.file_uploader(
    "📁 Upload Your Chest X-ray",
    type=['png', 'jpg', 'jpeg'],
    help="Drag and drop or click to upload a chest X-ray image (PNG, JPG, JPEG)",
    label_visibility="visible"
)

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    
    # Display uploaded image
    st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)
    
    # Make prediction
    if st.session_state.model is not None:
        with st.spinner("Analyzing image..."):
            result, confidence = predict_pneumonia(image)
        
        if result:
            # Display results
            if result == "NORMAL":
                st.markdown(f"""
                <div class="result-box">
                    <h2>✅ NORMAL</h2>
                    <p style="font-size: 1.2rem; margin: 1rem 0;">No signs of pneumonia detected in this X-ray.</p>
                    <p style="font-size: 1rem; color: #bdc3c7;">Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box pneumonia">
                    <h2>⚠️ PNEUMONIA DETECTED</h2>
                    <p style="font-size: 1.2rem; margin: 1rem 0;">Pneumonia indicators detected. Please consult a healthcare professional.</p>
                    <p style="font-size: 1rem; color: #bdc3c7;">Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence visualization
            confidence_percent = confidence * 100
            confidence_class = "high" if confidence_percent > 80 else ""
            
            st.markdown(f"""
            <div class="confidence-bar">
                <p style="margin-bottom: 0.5rem; font-weight: 600;">AI Confidence: {confidence_percent:.1f}%</p>
                <div class="confidence-fill {confidence_class}" style="width: {confidence_percent}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk level
            if result == "PNEUMONIA":
                risk_level = "High" if confidence > 0.9 else "Medium" if confidence > 0.75 else "Low"
                st.info(f"⚠️ Risk Level: {risk_level}")
        else:
            st.error("❌ Unable to process the image. Please try again.")
    else:
        st.error("❌ Model not loaded. Please refresh the page.")

# Footer with enhanced disclaimers
st.markdown("""
<div class="footer">
    <p> AI-Powered Pneumonia Detection | Built with TensorFlow & Streamlit</p>
    <p style="font-size: 0.9rem; color: #dc3545; font-weight: bold;">⚠️ DISCLAIMER: This is an AI demonstration tool for educational purposes only.</p>
    <p style="font-size: 0.85rem;"><em>Not intended for medical diagnosis. Always consult qualified healthcare professionals for medical decisions.</em></p>
    <p style="font-size: 0.8rem; color: #6c757d; margin-top: 1rem;">Model Accuracy: 85.1% | Precision: 95.4% | Recall: 80.0%</p>
</div>
""", unsafe_allow_html=True)