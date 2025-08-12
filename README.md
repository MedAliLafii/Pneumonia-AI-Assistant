# 🫁 AI Pneumonia Detection

A clean, simple deep learning application for detecting pneumonia from chest X-ray images using Convolutional Neural Networks (CNNs). This project demonstrates practical AI application development with a focus on user experience and deployment readiness.

## 🎯 Project Overview

This AI system achieves **85.1% accuracy** with **95.4% precision** in pneumonia detection, providing instant analysis of chest X-ray images. Built with modern technologies and designed for easy deployment.

## ✨ Key Features

- **🔮 Simple Upload**: Just drag and drop or click to upload X-ray images
- **⚡ Instant Results**: Get immediate AI-powered analysis with confidence scores
- **🎨 Clean Interface**: Modern, responsive design that works on all devices
- **🧠 Advanced CNN**: Sophisticated neural network with 423K+ parameters

## 🛠️ Technology Stack

### Core Technologies

- **TensorFlow/Keras**: Deep learning framework for CNN implementation
- **Streamlit**: Web application framework for rapid deployment
- **Python**: Primary programming language
- **PIL/Pillow**: Image processing and manipulation

### Data Science & Visualization

- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive charts and dashboards
- **Scikit-learn**: Machine learning utilities

### Medical AI Focus

- **Medical Imaging**: X-ray analysis and preprocessing
- **Data Augmentation**: Rotation, zoom, flip, brightness adjustments
- **Transfer Learning**: Leveraging pre-trained models
- **Model Deployment**: Production-ready ML pipeline

## 📊 Performance Metrics

| Metric          | Value | Description                       |
| --------------- | ----- | --------------------------------- |
| **Accuracy**    | 85.1% | Overall classification accuracy   |
| **Precision**   | 95.4% | Minimized false positives         |
| **Recall**      | 80.0% | Detected 80% of pneumonia cases   |
| **F1-Score**    | 87.0% | Balanced precision-recall measure |
| **Specificity** | 93.6% | Normal cases correctly identified |
| **AUC-ROC**     | 0.893 | Area under ROC curve              |

## 🏗️ Model Architecture

The CNN architecture features:

- **4 Convolutional Blocks**: Progressive feature extraction (32→64→128→256 filters)
- **Batch Normalization**: Stabilizes training and improves convergence
- **Global Average Pooling**: Reduces overfitting compared to flatten layer
- **Dropout Layers**: 50% and 30% dropout rates for regularization
- **Dense Layers**: 128 neurons with ReLU activation before final sigmoid

**Total Parameters**: 423,873 (422,593 trainable)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

## 📈 Model Training

The model was trained on a comprehensive dataset of chest X-ray images with:

- **Data Augmentation**: Rotation, zoom, flip, brightness adjustments
- **Regularization**: Dropout, batch normalization, early stopping
- **Optimization**: Adam optimizer with learning rate scheduling
- **Validation**: 20% holdout set for unbiased evaluation

## 🔬 Technical Details

### Data Preprocessing

- Image resizing to 224x224 pixels
- Grayscale conversion and normalization
- Real-time augmentation during training

### Model Training

- **Epochs**: 20 (with early stopping)
- **Batch Size**: 32
- **Learning Rate**: 0.001 (with reduction on plateau)
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

### Performance Optimization

- Model caching for faster inference
- Efficient image preprocessing pipeline
- Responsive UI for various screen sizes

## 🤝 Contributing

This project is designed as a portfolio piece, but contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request
