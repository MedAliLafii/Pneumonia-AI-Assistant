# AI Pneumonia Assistant - X-Ray Analysis

A deep learning system for chest X-ray pneumonia detection using CNN (Convolutional Neural Network).

## Features

- 🔬 **X-Ray Analysis**: Deep learning CNN for pneumonia detection
- 📊 **Real-time Results**: Instant prediction with confidence scores
- 🔗 **Cross-App Navigation**: Links to Medical Assistant chatbot
- 📱 **Responsive Design**: Works on desktop and mobile
- 🚀 **Production Ready**: Optimized for Vercel deployment

## Tech Stack

- **Backend**: Flask, Python
- **AI/ML**: TensorFlow CPU, Keras, CNN
- **Image Processing**: OpenCV, Pillow
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Deployment**: Vercel

## Setup

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set environment variables**:

   - Copy `env.example` to `.env`
   - Set cross-app URLs after deployment

4. **Run locally**:
   ```bash
   python app.py
   ```

## Deployment

### Vercel (Recommended)

1. **Deploy to Vercel**:

   ```bash
   vercel
   ```

2. **Set environment variables** in Vercel dashboard:
   - `CNN_APP_URL` (your Vercel URL)
   - `CHATBOT_APP_URL` (after chatbot app deployment)

## Environment Variables

- `CNN_APP_URL`: URL of this CNN app
- `CHATBOT_APP_URL`: URL of the Medical Assistant chatbot app

## API Endpoints

- `GET /`: Main X-Ray analysis interface
- `POST /upload_image`: Image upload and analysis
- `GET /health`: Health check

## Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Training Data**: Chest X-ray images (normal vs pneumonia)
- **Accuracy**: High accuracy for pneumonia detection
- **Input**: Chest X-ray images (JPG, PNG, etc.)
- **Output**: Prediction with confidence score

## Cross-App Integration

This app is designed to work with the Medical Assistant chatbot app. After deploying both apps:

1. Set `CHATBOT_APP_URL` to your chatbot app URL
2. Set `CNN_APP_URL` to this app's URL
3. The sidebar will provide navigation between both apps

## File Structure

```
cnn/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── models/            # CNN model files
├── src/               # Source code
├── templates/         # HTML templates
├── static/            # CSS, JS, images
├── vercel.json        # Vercel configuration
├── .vercelignore      # Vercel ignore file
└── README.md          # This file
```

## Disclaimer

This is an AI demonstration tool for educational purposes only. Not intended for medical diagnosis. Always consult qualified healthcare professionals.
