# AI Medical Assistant - Chatbot

A specialized chatbot for pneumonia-related medical information using Google Gemini and RAG (Retrieval-Augmented Generation).

## Features

- 🤖 **AI Medical Assistant**: Powered by Google Gemini 2.0 Flash
- 📚 **RAG System**: Retrieval-Augmented Generation with medical documents
- 🔗 **Cross-App Navigation**: Links to X-Ray Analysis app
- 📱 **Responsive Design**: Works on desktop and mobile
- ⚡ **Fast Loading**: Optimized for Vercel deployment

## Tech Stack

- **Backend**: Flask, Python
- **AI**: Google Gemini 2.0 Flash
- **RAG**: LangChain, FAISS, Sentence Transformers
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
   - Add your `GEMINI_API_KEY`
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
   - `GEMINI_API_KEY`
   - `CNN_APP_URL` (after CNN app deployment)
   - `CHATBOT_APP_URL` (your Vercel URL)

## Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key
- `CNN_APP_URL`: URL of the CNN X-Ray Analysis app
- `CHATBOT_APP_URL`: URL of this chatbot app

## API Endpoints

- `GET /`: Main chatbot interface
- `POST /get`: Chat endpoint
- `GET /health`: Health check

## Cross-App Integration

This app is designed to work with the CNN X-Ray Analysis app. After deploying both apps:

1. Set `CNN_APP_URL` to your CNN app URL
2. Set `CHATBOT_APP_URL` to this app's URL
3. The sidebar will provide navigation between both apps

## Disclaimer

This is an AI demonstration tool for educational purposes only. Not intended for medical diagnosis. Always consult qualified healthcare professionals.
