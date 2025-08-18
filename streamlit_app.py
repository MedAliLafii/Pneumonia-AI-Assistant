import streamlit as st
import os
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt
from src.cnn_predictor import MediscopePredictor
import uuid
from PIL import Image
import io

# Import deployment configuration
from deployment_config import configure_deployment_environment

# Configure deployment environment
configure_deployment_environment()

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
else:
    st.warning("⚠️ GEMINI_API_KEY not found. Chatbot functionality will be limited.")

# Page configuration
st.set_page_config(
    page_title="Mediscope AI - Pneumonia Detection & Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #666;
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .success-box {
        background-color: #0d6efd;
        border: 4px solid #0a58ca;
        border-radius: 0.5rem;
        padding: 2.5rem;
        margin: 1rem 0;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .error-box {
        background-color: #f8d7da;
        border: 2px solid #721c24;
        border-radius: 0.5rem;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        color: #721c24;
        font-weight: bold;
    }
    .centered-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stButton > button {
        border-radius: 0.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Reduce sidebar spacing */
    .css-1d391kg {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    .css-1lcbmhc {
        padding: 0.5rem !important;
    }
    
    /* Reduce margins in sidebar */
    .css-1d391kg .stMarkdown {
        margin-bottom: 0.5rem !important;
    }
    
    /* Target the sidebar header specifically */
    .st-emotion-cache-kgpedg {
        padding: 0.25rem !important;
        margin: 0 !important;
    }
    
    /* Reduce spacing in sidebar header */
    .st-emotion-cache-kgpedg .st-emotion-cache-11ukie {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Reduce overall sidebar padding */
    .css-1d391kg {
        padding: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'mediscope_predictor' not in st.session_state:
    st.session_state.mediscope_predictor = None
if 'docsearch' not in st.session_state:
    st.session_state.docsearch = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if 'conversation_start_time' not in st.session_state:
    st.session_state.conversation_start_time = None

def initialize_components():
    """Initialize components at startup to avoid cold start issues"""
    if st.session_state.initialized:
        return
    
    # Load embeddings
    if st.session_state.embeddings is None:
        st.session_state.embeddings = download_hugging_face_embeddings()
    
    # Load CNN model
    if st.session_state.mediscope_predictor is None:
        st.session_state.mediscope_predictor = MediscopePredictor()
    
    # Load FAISS index
    if st.session_state.docsearch is None:
        try:
            st.session_state.docsearch = FAISS.load_local("faiss_index", st.session_state.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            try:
                data = "data"
                documents = load_pdf_file(data)
                text_chunks = text_split(documents)
                st.session_state.docsearch = FAISS.from_documents(text_chunks, st.session_state.embeddings)
            except Exception as e2:
                # Create a minimal fallback
                from langchain_core.documents import Document
                st.session_state.docsearch = FAISS.from_documents([Document(page_content="Pneumonia information not available")], st.session_state.embeddings)
    
    # Initialize RAG chain
    if st.session_state.rag_chain is None and st.session_state.docsearch is not None and GEMINI_API_KEY:
        try:
            retriever = st.session_state.docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
            chatModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
            st.session_state.rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        except Exception as e:
            pass  # Silently fail, chatbot will be unavailable
    
    st.session_state.initialized = True

def cnn_page():
    """CNN Pneumonia Detection Page"""
    st.markdown('<h1 class="main-header">🔬 Pneumonia Detection</h1>', unsafe_allow_html=True)
    
    # Center the content
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📸 Upload a Chest X-Ray Image
    This AI-powered tool can analyze chest X-ray images to detect signs of pneumonia. 
    Please upload a clear, high-quality chest X-ray image in JPG, PNG, or similar format.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chest X-ray image for pneumonia detection (PNG, JPG, JPEG only)"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Check file size (max 10MB)
        file_size = len(uploaded_file.getvalue())
        if file_size > 10 * 1024 * 1024:  # 10MB
            st.error("❌ File too large. Please upload an image smaller than 10MB.")
            return
        
        st.subheader("📸 Uploaded Image")
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray Image", use_container_width=True, width=400)
        except Exception as e:
            st.error(f"❌ Failed to display image: {str(e)}")
            return
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("🔍 Analysis Results")
        
        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                if st.session_state.mediscope_predictor is None:
                    st.error("❌ Model not available. Please try again later.")
                else:
                    with st.spinner("Analyzing image..."):
                        try:
                            # Save uploaded file to temporary location
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                uploaded_file.seek(0)
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            # Verify the file was saved correctly
                            if not os.path.exists(tmp_file_path):
                                st.error("❌ Failed to save uploaded file")
                                return
                            
                            # Make prediction
                            if st.session_state.mediscope_predictor.model is None:
                                st.error("❌ Model not properly loaded. Please try again later.")
                                return
                            
                            result = st.session_state.mediscope_predictor.predict(tmp_file_path)
                            
                            # Clean up temporary file
                            try:
                                os.unlink(tmp_file_path)
                            except:
                                pass
                            
                            # Display results
                            if result.get("error"):
                                st.error(f"❌ Error: {result['error']}")
                            else:
                                prediction = result["prediction"]
                                confidence = result["confidence_percentage"]
                                
                                if prediction == "Pneumonia":
                                    st.markdown(f"<h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>⚠️ PNEUMONIA DETECTED</h1>", unsafe_allow_html=True)
                                    st.markdown(f"<h2 style='font-size: 1.8rem; margin-bottom: 1rem;'>Confidence: {confidence}</h2>", unsafe_allow_html=True)
                                    st.markdown("<p style='font-size: 1.2rem;'>Please consult a healthcare professional immediately.</p>", unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>✅ NORMAL CHEST X-RAY</h1>", unsafe_allow_html=True)
                                    st.markdown(f"<h2 style='font-size: 1.8rem; margin-bottom: 1rem;'>Confidence: {confidence}</h2>", unsafe_allow_html=True)
                                    st.markdown("<p style='font-size: 1.2rem;'>No signs of pneumonia detected.</p>", unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Show detailed scores
                                st.markdown('<h3 style="text-align: center; margin-top: 2rem;">📊 Detailed Analysis</h3>', unsafe_allow_html=True)
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Pneumonia Score", f"{result['class_scores']['pneumonia']:.1f}%")
                                with col_b:
                                    st.metric("Normal Score", f"{result['class_scores']['normal']:.1f}%")
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Summary
                                summary = st.session_state.mediscope_predictor.get_prediction_summary(result)
                                st.markdown(f'<div style="text-align: center; padding: 1.5rem; background-color: #2c3e50; border-radius: 0.5rem; border: 2px solid #34495e; color: white; font-weight: bold; font-size: 1.1rem; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">{summary}</div>', unsafe_allow_html=True)
                                 
                        except Exception as e:
                            st.error(f"❌ Processing failed: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Close centered content div
    st.markdown('</div>', unsafe_allow_html=True)

def chatbot_page():
    """Medical Assistant Chatbot Page"""
    st.markdown('<h1 class="main-header">💬 Medical Assistant</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Ask Questions About Pneumonia
    I'm Mediscope AI, your specialized assistant for pneumonia-related questions. 
    I can provide information about symptoms, prevention, and general medical knowledge about pneumonia.
    """)
    
    # Check if RAG chain is available
    if st.session_state.rag_chain is None:
        st.error("❌ Medical assistant is not available at the moment. Please try again later.")
        return
    
    # Chat interface
    st.subheader("💬 Chat with Mediscope AI")
    
    # Initialize conversation start time if not set
    if st.session_state.conversation_start_time is None:
        st.session_state.conversation_start_time = datetime.now()
    
    # Display conversation info
    if st.session_state.chat_history:
        conversation_duration = datetime.now() - st.session_state.conversation_start_time
        st.caption(f"💬 Conversation ID: {st.session_state.conversation_id[:8]}... | Duration: {conversation_duration.seconds//60}m {conversation_duration.seconds%60}s | Messages: {len(st.session_state.chat_history)}")
    
    # Display chat history with timestamps
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "timestamp" in message:
                st.caption(f"🕐 {message['timestamp']}")
    
    # Chat input
    if prompt := st.chat_input("Ask me about pneumonia..."):
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Add user message to chat history with timestamp
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": current_time,
            "message_id": len(st.session_state.chat_history)
        }
        st.session_state.chat_history.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
            st.caption(f"🕐 {current_time}")
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get conversation context
                    chat_context = get_chat_context()
                    
                    # Create enhanced prompt with context
                    enhanced_prompt = f"{prompt}\n\n{chat_context}" if chat_context else prompt
                    
                    # Add conversation history to the input
                    response = st.session_state.rag_chain.invoke({
                        "input": enhanced_prompt,
                        "conversation_history": chat_context
                    })
                    ai_response = response["answer"]
                    
                    # Add AI response to chat history with timestamp
                    ai_message = {
                        "role": "assistant", 
                        "content": ai_response,
                        "timestamp": current_time,
                        "message_id": len(st.session_state.chat_history)
                    }
                    st.session_state.chat_history.append(ai_message)
                    
                    st.write(ai_response)
                    st.caption(f"🕐 {current_time}")
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    error_message = {
                        "role": "assistant", 
                        "content": error_msg,
                        "timestamp": current_time,
                        "message_id": len(st.session_state.chat_history)
                    }
                    st.session_state.chat_history.append(error_message)
                    st.error(error_msg)
                    st.caption(f"🕐 {current_time}")
    
def get_chat_context():
    """Get conversation context for the AI"""
    if not st.session_state.chat_history:
        return ""
    
    context = "\n\nCONVERSATION HISTORY:\n"
    for message in st.session_state.chat_history[-6:]:  # Last 6 messages for context
        role = "User" if message["role"] == "user" else "Mediscope AI"
        context += f"{role}: {message['content']}\n"
    
    return context

def export_chat_history():
    """Export chat history to a downloadable file"""
    if not st.session_state.chat_history:
        st.warning("No chat history to export")
        return
    
    # Create export content
    export_content = f"""Mediscope AI - Chat History Export
Conversation ID: {st.session_state.conversation_id}
Start Time: {st.session_state.conversation_start_time.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.conversation_start_time else 'N/A'}
Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Messages: {len(st.session_state.chat_history)}

{'='*50}

"""
    
    for i, message in enumerate(st.session_state.chat_history):
        role = "👤 User" if message["role"] == "user" else "🤖 Mediscope AI"
        timestamp = message.get("timestamp", "N/A")
        content = message["content"]
        
        export_content += f"[{timestamp}] {role}:\n{content}\n\n"
    
    # Create download button
    st.download_button(
        label="📄 Download Chat History",
        data=export_content,
        file_name=f"mediscope_chat_{st.session_state.conversation_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )

def main():
    """Main application"""
    # Initialize components if not already done
    if not st.session_state.initialized:
        # Show loading screen
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 70vh;">
            <div style="text-align: center;">
                <h1 style="color: #1f77b4; margin-bottom: 2rem;">🏥 Mediscope AI</h1>
                <div style="font-size: 2rem; margin-bottom: 1rem;">⏳</div>
                <p style="font-size: 1.2rem; color: #666;">Loading application...</p>
                <p style="font-size: 1rem; color: #888;">Please wait while we initialize the AI models</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize components
        initialize_components()
        st.rerun()
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2>🏥 Mediscope AI</h2>', unsafe_allow_html=True)
        st.markdown("*Advanced Pneumonia Detection & Medical Assistant*")
        
        st.markdown("---")
        
        # Navigation buttons
        st.markdown("**📋 Navigation**")
        
        if st.button("🔬 Detection", use_container_width=True):
            st.session_state.current_page = "detection"
        
        if st.button("💬 Assistant", use_container_width=True):
            st.session_state.current_page = "assistant"
        
        # Set default page if not set
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "detection"
        
        st.markdown("---")
        
        # Disclaimer
        st.markdown("""
        **⚠️ Medical Disclaimer**
        
        This application is for educational purposes only. It is not a substitute for professional medical advice. Always consult healthcare professionals for medical concerns.
        """, help="Medical disclaimer")
    
    # Page routing
    if st.session_state.current_page == "detection":
        cnn_page()
    elif st.session_state.current_page == "assistant":
        chatbot_page()

if __name__ == "__main__":
    main()
