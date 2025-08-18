from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document
import streamlit as st
import numpy as np
import os
import time


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks


class SimpleFallbackEmbeddings:
    """
    A simple fallback embeddings class that generates random embeddings.
    This is used when HuggingFace embeddings fail to load.
    """
    def __init__(self, dimension=384):
        self.dimension = dimension
    
    def embed_documents(self, texts):
        """Generate random embeddings for documents."""
        embeddings = []
        for text in texts:
            # Generate a deterministic embedding based on text length and content
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.normal(0, 1, self.dimension)
            embeddings.append(embedding.tolist())
        return embeddings
    
    def embed_query(self, text):
        """Generate random embedding for a query."""
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.normal(0, 1, self.dimension)
        return embedding.tolist()


#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    """
    Download HuggingFace embeddings with enhanced error handling and fallback options.
    """
    # Set environment variables to reduce rate limiting issues
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Try multiple models in order of preference
    models_to_try = [
        'sentence-transformers/all-MiniLM-L6-v2',  # 384 dimensions, smaller
        'sentence-transformers/paraphrase-MiniLM-L3-v2',  # 384 dimensions, even smaller
        'sentence-transformers/all-mpnet-base-v2'  # 768 dimensions, fallback
    ]
    
    for model_name in models_to_try:
        try:
            st.info(f"🔄 Attempting to load embeddings model: {model_name}")
            
            # Add retry logic with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    embeddings = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs={'device': 'cpu'},  # Force CPU usage
                        encode_kwargs={'normalize_embeddings': True}
                    )
                    
                    # Test the embeddings with a simple query
                    test_embedding = embeddings.embed_query("test")
                    if len(test_embedding) > 0:
                        st.success(f"✅ HuggingFace embeddings loaded successfully: {model_name}")
                        return embeddings
                    else:
                        raise Exception("Embeddings test failed")
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        st.warning(f"⚠️ Attempt {attempt + 1} failed for {model_name}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        st.warning(f"⚠️ Failed to load {model_name} after {max_retries} attempts: {str(e)}")
                        break
                        
        except Exception as e:
            st.warning(f"⚠️ Error with model {model_name}: {str(e)}")
            continue
    
    # If all models fail, use fallback
    st.warning("⚠️ All HuggingFace models failed. Using fallback embeddings (limited functionality)")
    return SimpleFallbackEmbeddings()