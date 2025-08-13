from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Load and process the PDF data
print("Loading PDF data...")
extracted_data = load_pdf_file(data='data/')
print(f"Loaded {len(extracted_data)} documents")

print("Filtering documents...")
filter_data = filter_to_minimal_docs(extracted_data)
print(f"Filtered to {len(filter_data)} documents")

print("Splitting text into chunks...")
text_chunks = text_split(filter_data)
print(f"Created {len(text_chunks)} text chunks")

print("Downloading embeddings...")
embeddings = download_hugging_face_embeddings()

print("Creating FAISS vector store...")
# Create FAISS vector store
docsearch = FAISS.from_documents(text_chunks, embeddings)

print("Saving vector store...")
# Save the vector store
docsearch.save_local("faiss_index")

print("✅ Vector store created and saved successfully!")
print(f"Total documents indexed: {len(text_chunks)}")