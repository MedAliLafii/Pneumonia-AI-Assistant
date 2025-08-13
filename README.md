# AI Pneumonia Assistant

A comprehensive AI-powered medical assistant split into two specialized applications for optimal deployment and performance.

## 🏗️ Project Structure

This project has been split into two separate applications:

### 🤖 **Chatbot App** (`chatbot/`)
- **Purpose**: Medical consultation and information retrieval
- **Features**: RAG-powered chatbot using Google Gemini 2.0 Flash
- **Deployment**: Vercel (lightweight, fast)
- **Tech Stack**: Flask, LangChain, FAISS, Google Gemini

### 🔬 **CNN App** (`cnn/`)
- **Purpose**: X-ray analysis and pneumonia detection
- **Features**: Deep learning CNN for chest X-ray classification
- **Deployment**: Vercel (CPU-optimized)
- **Tech Stack**: Flask, TensorFlow CPU, OpenCV, CNN

## 🚀 Quick Start

### Chatbot App
```bash
cd chatbot
pip install -r requirements.txt
python app.py
```

### CNN App
```bash
cd cnn
pip install -r requirements.txt
python app.py
```

## 📁 Directory Structure

```
Pneumonia/
├── chatbot/                 # Medical Assistant Chatbot
│   ├── app.py              # Flask application
│   ├── requirements.txt    # Dependencies
│   ├── src/                # Source code
│   ├── templates/          # HTML templates
│   ├── static/             # CSS, JS files
│   ├── data/               # PDF medical data
│   ├── faiss_index/        # Vector database
│   ├── vercel.json         # Vercel configuration
│   └── README.md           # Chatbot documentation
├── cnn/                    # X-Ray Analysis System
│   ├── app.py              # Flask application
│   ├── requirements.txt    # Dependencies
│   ├── src/                # Source code
│   ├── templates/          # HTML templates
│   ├── static/             # CSS, JS files
│   ├── models/             # CNN model files
│   ├── vercel.json         # Vercel configuration
│   └── README.md           # CNN documentation
└── README.md               # This file
```

## 🔗 Cross-App Integration

Both apps are designed to work together:
- **Sidebar Navigation**: Each app has links to the other
- **Environment Variables**: Configure URLs for cross-linking
- **Unified Experience**: Seamless user experience across both apps

## 🛠️ Deployment

### Chatbot App (Vercel)
1. Deploy to Vercel: `vercel`
2. Set environment variables:
   - `GEMINI_API_KEY`
   - `CNN_APP_URL`
   - `CHATBOT_APP_URL`

### CNN App (Vercel)
1. Deploy to Vercel: `vercel`
2. Set environment variables:
   - `CNN_APP_URL`
   - `CHATBOT_APP_URL`

## 🎯 Features

### Chatbot App
- 🤖 **RAG-powered responses** using medical knowledge base
- 📚 **PDF document processing** for medical information
- 🔍 **Semantic search** with FAISS vector database
- 💬 **Natural language interaction** with Google Gemini 2.0 Flash

### CNN App
- 🔬 **X-ray analysis** with deep learning CNN
- 📊 **Real-time predictions** with confidence scores
- 🖼️ **Image processing** and validation
- 📱 **Responsive interface** for mobile and desktop

## 🔧 Tech Stack

### Backend
- **Flask**: Web framework
- **Python**: Core language
- **LangChain**: RAG framework
- **TensorFlow**: Deep learning (CNN app)

### AI/ML
- **Google Gemini 2.0 Flash**: Chat model
- **FAISS**: Vector database
- **CNN**: X-ray classification
- **HuggingFace**: Embeddings

### Frontend
- **HTML/CSS/JavaScript**: Custom styling
- **Bootstrap**: Responsive design
- **Jinja2**: Template engine

### Deployment
- **Vercel**: Serverless deployment
- **Environment Variables**: Configuration management

## 📋 Environment Variables

### Chatbot App
- `GEMINI_API_KEY`: Google Gemini API key
- `CNN_APP_URL`: URL of CNN app
- `CHATBOT_APP_URL`: URL of chatbot app

### CNN App
- `CNN_APP_URL`: URL of CNN app
- `CHATBOT_APP_URL`: URL of chatbot app

## 🚨 Disclaimer

This is an AI demonstration tool for educational purposes only. Not intended for medical diagnosis. Always consult qualified healthcare professionals for medical advice.

## 📄 License

This project is for educational and demonstration purposes.
