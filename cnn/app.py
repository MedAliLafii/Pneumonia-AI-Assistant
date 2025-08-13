from flask import Flask, render_template, jsonify, request, redirect, url_for
from src.cnn_predictor import PneumoniaPredictor
import os
import uuid
from werkzeug.utils import secure_filename
import tempfile
from dotenv import load_dotenv

app = Flask(__name__)

# Configure upload settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

load_dotenv()

# Environment variables for cross-linking
CNN_APP_URL = os.getenv('CNN_APP_URL', 'http://localhost:5001')
CHATBOT_APP_URL = os.getenv('CHATBOT_APP_URL', 'http://localhost:5000')

# Initialize components (will be lazy loaded)
pneumonia_predictor = None

def initialize_components():
    """Initialize components lazily to avoid cold start issues"""
    global pneumonia_predictor
    
    if pneumonia_predictor is None:
        pneumonia_predictor = PneumoniaPredictor()

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    """Main route - CNN page"""
    return render_template('cnn_page.html', 
                         cnn_url=CNN_APP_URL, 
                         chatbot_url=CHATBOT_APP_URL)

@app.route("/upload_image", methods=["POST"])
def upload_image():
    """Handle image upload for pneumonia detection."""
    initialize_components()
    if pneumonia_predictor is None:
        return jsonify({"error": "Model not available"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        # Use temporary file for Vercel compatibility
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            try:
                # Save the file to temporary location
                file.save(tmp_file.name)
                
                # Make prediction
                result = pneumonia_predictor.predict(tmp_file.name)
                summary = pneumonia_predictor.get_prediction_summary(result)
                
                return jsonify({
                    "success": True,
                    "prediction": result,
                    "summary": summary
                })
                
            except Exception as e:
                return jsonify({"error": f"Processing failed: {str(e)}"}), 500
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route("/health")
def health_check():
    """Health check endpoint to verify all components are working."""
    initialize_components()
    status = {
        "cnn_model": "✅ Ready" if pneumonia_predictor is not None and pneumonia_predictor.model is not None else "❌ Not loaded",
        "cnn_url": CNN_APP_URL,
        "chatbot_url": CHATBOT_APP_URL
    }
    return jsonify(status)

# For Vercel deployment
app.debug = False

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
