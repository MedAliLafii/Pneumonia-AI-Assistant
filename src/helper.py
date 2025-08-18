from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document
import streamlit as st
import numpy as np


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
    Download HuggingFace embeddings with error handling and fallback options.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
        st.success("✅ HuggingFace embeddings loaded successfully")
        return embeddings
        
    except ImportError as e:
        st.error(f"❌ ImportError: {str(e)}")
        st.error("Missing required dependencies. Please ensure torch, tokenizers, safetensors, and huggingface-hub are installed.")
        st.warning("⚠️ Using fallback embeddings (limited functionality)")
        return SimpleFallbackEmbeddings()
        
    except Exception as e:
        st.warning(f"⚠️ Error loading embeddings: {str(e)}")
        st.warning("⚠️ Using fallback embeddings (limited functionality)")
        return SimpleFallbackEmbeddings()