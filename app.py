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
import tempfile

app = Flask(__name__)

# Configure upload settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

load_dotenv()

GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Initialize components (will be lazy loaded)
embeddings = None
pneumonia_predictor = None
docsearch = None
rag_chain = None

def initialize_components():
    """Initialize components lazily to avoid cold start issues"""
    global embeddings, pneumonia_predictor, docsearch, rag_chain
    
    if embeddings is None:
        embeddings = download_hugging_face_embeddings()
    
    if pneumonia_predictor is None:
        pneumonia_predictor = PneumoniaPredictor()
    
    if docsearch is None:
        # For Vercel deployment, we'll need to handle this differently
        # For now, we'll create a simple fallback
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
                docsearch = FAISS.from_documents([Document(page_content="Medical information not available")], embeddings)
    
    if rag_chain is None and docsearch is not None:
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
    initialize_components()
    if rag_chain is None:
        return "Chatbot is not available at the moment. Please try again later."
    
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
        "chatbot": "✅ Ready" if rag_chain is not None else "❌ Not loaded",
        "cnn_model": "✅ Ready" if pneumonia_predictor is not None and pneumonia_predictor.model is not None else "❌ Not loaded",
        "faiss_index": "✅ Ready" if docsearch is not None else "❌ Not loaded"
    }
    return jsonify(status)

# For Vercel deployment
app.debug = False

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
