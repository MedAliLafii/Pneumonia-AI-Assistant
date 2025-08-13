from flask import Flask, render_template, jsonify, request, redirect, url_for
from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from src.cnn_predictor import PneumoniaPredictor
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

load_dotenv()

GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Initialize components
embeddings = download_hugging_face_embeddings()
pneumonia_predictor = PneumoniaPredictor()

# Load the saved FAISS vector store
try:
    docsearch = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("✅ Loaded existing FAISS index")
except Exception as e:
    print(f"⚠️  Error loading index: {e}")
    print("Creating new index from PDF data...")
    # Create FAISS vector store from scratch if no saved index exists
    data = "data"
    documents = load_pdf_file(data)
    text_chunks = text_split(documents)
    docsearch = FAISS.from_documents(text_chunks, embeddings)
    print("✅ Created new FAISS index")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    """Main route - redirect to CNN page"""
    return redirect(url_for('cnn_page'))

@app.route("/cnn")
def cnn_page():
    """CNN X-Ray Analysis page"""
    return render_template('cnn_page.html')

@app.route("/chatbot")
def chatbot_page():
    """Chatbot Medical Assistant page"""
    return render_template('chatbot_page.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

@app.route("/upload_image", methods=["POST"])
def upload_image():
    """Handle image upload for pneumonia detection."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            # Save the file
            file.save(filepath)
            
            # Make prediction
            result = pneumonia_predictor.predict(filepath)
            summary = pneumonia_predictor.get_prediction_summary(result)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                "success": True,
                "prediction": result,
                "summary": summary
            })
            
        except Exception as e:
            # Clean up file if it exists
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route("/health")
def health_check():
    """Health check endpoint to verify all components are working."""
    status = {
        "chatbot": "✅ Ready",
        "cnn_model": "✅ Ready" if pneumonia_predictor.model is not None else "❌ Not loaded",
        "faiss_index": "✅ Ready" if docsearch is not None else "❌ Not loaded"
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
