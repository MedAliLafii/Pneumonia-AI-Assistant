from flask import Flask, render_template, jsonify, request, redirect, url_for
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Environment variables for cross-linking
CNN_APP_URL = os.getenv('CNN_APP_URL', 'http://localhost:5001')
CHATBOT_APP_URL = os.getenv('CHATBOT_APP_URL', 'http://localhost:5000')

# Initialize components
embeddings = None
docsearch = None
rag_chain = None

def initialize_components():
    """Initialize components with error handling"""
    global embeddings, docsearch, rag_chain
    
    try:
        if embeddings is None:
            print("🔄 Loading embeddings...")
            embeddings = download_hugging_face_embeddings()
            print("✅ Embeddings loaded")
        
        if docsearch is None:
            # Only try to load existing FAISS index, never create new one
            try:
                print("🔄 Loading FAISS index...")
                docsearch = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                print("✅ Loaded existing FAISS index")
            except Exception as e:
                print(f"❌ No existing FAISS index found: {e}")
                # Create a minimal fallback with basic medical information
                from langchain_core.documents import Document
                fallback_content = """
                Pneumonia is an infection that inflames the air sacs in one or both lungs. 
                The air sacs may fill with fluid or pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing.
                
                Common symptoms include:
                - Cough with phlegm or pus
                - Fever, sweating and shaking chills
                - Shortness of breath
                - Rapid, shallow breathing
                - Sharp or stabbing chest pain that gets worse when you breathe deeply or cough
                - Loss of appetite, low energy, and fatigue
                - Nausea and vomiting, especially in small children
                - Confusion, especially in older people
                
                Risk factors include:
                - Age (children under 2 and adults over 65)
                - Smoking
                - Chronic diseases (asthma, COPD, heart disease)
                - Weakened immune system
                - Recent surgery or hospitalization
                
                Treatment typically involves antibiotics for bacterial pneumonia, rest, and supportive care.
                """
                docsearch = FAISS.from_documents([Document(page_content=fallback_content)], embeddings)
                print("✅ Created fallback FAISS index")
        
        if rag_chain is None and docsearch is not None:
            print("🔄 Setting up RAG chain...")
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
            print("✅ RAG chain ready")
            
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()

# Initialize components lazily to avoid serverless crashes
print("🚀 AI Medical Assistant ready for lazy initialization!")

@app.route("/")
def index():
    """Main route - chatbot page"""
    return render_template('chatbot_page.html', 
                         cnn_url=CNN_APP_URL, 
                         chatbot_url=CHATBOT_APP_URL)

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        # Initialize components lazily
        initialize_components()
        
        if rag_chain is None:
            return "Chatbot is not available at the moment. Please try again later."
        
        if not GEMINI_API_KEY:
            return "API key not configured. Please check your environment variables."
        
        msg = request.form["msg"]
        input = msg
        print(f"User input: {input}")
        
        response = rag_chain.invoke({"input": msg})
        print("Response : ", response["answer"])
        return str(response["answer"])
    except Exception as e:
        print(f"Error in chat: {e}")
        import traceback
        traceback.print_exc()
        return "I'm sorry, I encountered an error. Please try again."

@app.route("/health")
def health_check():
    """Health check endpoint to verify all components are working."""
    try:
        # Initialize components lazily
        initialize_components()
        
        status = {
            "chatbot": "✅ Ready" if rag_chain is not None else "❌ Not loaded",
            "faiss_index": "✅ Ready" if docsearch is not None else "❌ Not loaded",
            "embeddings": "✅ Ready" if embeddings is not None else "❌ Not loaded",
            "api_key": "✅ Configured" if GEMINI_API_KEY else "❌ Missing",
            "cnn_url": CNN_APP_URL,
            "chatbot_url": CHATBOT_APP_URL
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "❌ Failed to initialize"
        }), 500

# For Vercel deployment
app.debug = False

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
