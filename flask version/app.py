from flask import Flask, render_template, jsonify, request, redirect, url_for
from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from src.cnn_predictor import MediscopePredictor
import os
import uuid
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)

# Configure upload settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

load_dotenv()

GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Global component variables
embeddings = None
mediscope_predictor = None
docsearch = None
rag_chain = None

def initialize_components():
    """Initialize components at startup to avoid cold start issues"""
    global embeddings, mediscope_predictor, docsearch, rag_chain
    
    print("🚀 Initializing Mediscope AI application...")
    
    # Load embeddings
    if embeddings is None:
        print("📥 Loading Hugging Face embeddings...")
        embeddings = download_hugging_face_embeddings()
        print("✅ Embeddings loaded successfully")
    
    # Load CNN model
    if mediscope_predictor is None:
        print("🤖 Loading pneumonia detection model...")
        mediscope_predictor = MediscopePredictor()
        print("✅ Pneumonia detection model loaded successfully")
    
    # Load FAISS index
    if docsearch is None:
        print("🔍 Loading FAISS index...")
        try:
            docsearch = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            print("✅ Loaded existing FAISS index")
        except Exception as e:
            print(f"⚠️  Error loading index: {e}")
            print("Creating new index from PDF data...")
            try:
                data = "data"
                documents = load_pdf_file(data)
                text_chunks = text_split(documents)
                docsearch = FAISS.from_documents(text_chunks, embeddings)
                print("✅ Created new FAISS index")
            except Exception as e2:
                print(f"❌ Failed to create index: {e2}")
                # Create a minimal fallback
                from langchain_core.documents import Document
                docsearch = FAISS.from_documents([Document(page_content="Pneumonia information not available")], embeddings)
                print("✅ Created fallback FAISS index")
    
    # Initialize RAG chain
    if rag_chain is None and docsearch is not None:
        print("🔗 Initializing RAG chain...")
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
        print("✅ RAG chain initialized successfully")
    
    print("🎉 All components initialized successfully!")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    """Main route - redirect to pneumonia detection page"""
    return redirect(url_for('pneumonia_detection'))

@app.route("/detection")
def pneumonia_detection():
    """Mediscope AI Pneumonia Detection page"""
    return render_template('pneumonia_detection.html')

@app.route("/assistant")
def medical_assistant():
    """Mediscope AI Pneumonia Medical Assistant page"""
    return render_template('medical_assistant.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    if rag_chain is None:
        return "Medical assistant is not available at the moment. Please try again later."
    
    msg = request.form["msg"]
    input = msg
    print(input)
    try:
        response = rag_chain.invoke({"input": msg})
        print("Response : ", response["answer"])
        return str(response["answer"])
    except Exception as e:
        print(f"Error in chat: {e}")
        return "I'm sorry, I encountered an error. Please try again."

@app.route("/upload_image", methods=["POST"])
def upload_image():
    """Handle image upload for pneumonia detection."""
    if mediscope_predictor is None:
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
                result = mediscope_predictor.predict(tmp_file.name)
                summary = mediscope_predictor.get_prediction_summary(result)
                
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
    status = {
        "medical_assistant": "✅ Ready" if rag_chain is not None else "❌ Not loaded",
        "pneumonia_model": "✅ Ready" if mediscope_predictor is not None and mediscope_predictor.model is not None else "❌ Not loaded",
        "faiss_index": "✅ Ready" if docsearch is not None else "❌ Not loaded"
    }
    return jsonify(status)

# For Vercel deployment
app.debug = False

# Initialize components at startup
if __name__ == '__main__':
    print("🚀 Starting Mediscope AI Pneumonia Detection System...")
    initialize_components()
    app.run(host="0.0.0.0", port=8080, debug=True)