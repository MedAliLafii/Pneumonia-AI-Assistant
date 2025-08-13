# Pneumonia AI Assistant

A comprehensive medical AI assistant that combines natural language processing and computer vision to provide both text-based answers about pneumonia and chest X-ray analysis for pneumonia detection.

## Features

### 🤖 AI Chatbot

- **Specialized Knowledge**: Trained specifically on pneumonia-related medical information
- **AI-Powered Responses**: Uses Google's Gemini AI model for intelligent responses
- **Document-Based Knowledge**: Built on a comprehensive PDF knowledge base
- **Real-time Responses**: Instant answers to medical questions about pneumonia

### 🔬 X-Ray Analysis

- **CNN-Based Detection**: Uses a trained convolutional neural network for pneumonia detection
- **Image Processing**: Supports multiple image formats (PNG, JPG, JPEG, GIF, BMP, TIFF)
- **Confidence Scoring**: Provides confidence percentages for predictions
- **Medical Disclaimer**: Includes appropriate medical disclaimers for AI-assisted diagnosis

### 🌐 Web Interface

- **Unified Interface**: Single application with tabbed interface for both functionalities
- **Modern Design**: Clean, responsive design with Bootstrap
- **Drag & Drop**: Easy image upload with drag-and-drop functionality
- **Real-time Feedback**: Loading indicators and clear result display

## Technology Stack

- **Backend**: Flask (Python)
- **AI Models**:
  - Google Gemini 2.0 Flash (for chatbot)
  - Custom CNN (for X-ray analysis)
- **Deep Learning**: TensorFlow/Keras
- **Vector Database**: FAISS for efficient document retrieval
- **Embeddings**: HuggingFace sentence-transformers
- **Image Processing**: OpenCV, Pillow
- **Frontend**: HTML, CSS, JavaScript with Bootstrap

## Setup

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file with your Google Gemini API key:

   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Knowledge Base Setup**:
   Place your pneumonia-related PDF documents in the `data/` folder, then run:

   ```bash
   python store_index.py
   ```

4. **Run the Application**:

   ```bash
   python app.py
   ```

5. **Access the Application**:
   Open your browser and go to `http://localhost:8080`

## Usage

### Chat Mode

- Switch to the "Chat" tab
- Ask questions about pneumonia symptoms, diagnosis, treatment, etc.
- Get instant AI-powered responses based on medical literature

### X-Ray Analysis Mode

- Switch to the "X-Ray Analysis" tab
- Upload a chest X-ray image (drag & drop or click to browse)
- Receive AI analysis with confidence scores
- Get appropriate medical disclaimers

## Project Structure

```
├── app.py                    # Main Flask application
├── store_index.py           # Script to build the knowledge base
├── src/
│   ├── helper.py            # Utility functions for PDF processing
│   ├── prompt.py            # AI system prompts
│   └── cnn_predictor.py     # CNN-based pneumonia detection
├── models/
│   └── improved_pneumonia_cnn.h5  # Trained CNN model
├── data/                    # PDF documents for knowledge base
├── uploads/                 # Temporary upload directory (auto-created)
├── templates/               # HTML templates
├── static/                  # CSS and static assets
└── faiss_index/             # Vector database (generated)
```

## API Endpoints

- `GET /` - Main application interface
- `POST /get` - Chatbot endpoint for text queries
- `POST /upload_image` - X-ray analysis endpoint
- `GET /health` - Health check endpoint

## Important Notes

⚠️ **Medical Disclaimer**: This application is for educational and research purposes only. The AI analysis should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Mohamed Ali Lafi - mohamedali.lafi@gmail.com
