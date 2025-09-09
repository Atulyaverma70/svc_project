import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PDFMinerLoader
import pickle

KNOWLEDGE_DIR = "knowledge_docs"

def load_documents():
    """Load all text and PDF documents"""
    docs = []
    # TXT files
    for folder in ["faqs", "troubleshooting"]:
        path = os.path.join(KNOWLEDGE_DIR, folder)
        for file in os.listdir(path):
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(path, file))
                docs.extend(loader.load())
    # PDF files
    path = os.path.join(KNOWLEDGE_DIR, "manuals")
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            loader = PDFMinerLoader(os.path.join(path, file))
            docs.extend(loader.load())
    return docs

def create_embeddings():
    """Create vector embeddings and save locally"""
    docs = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    # Save embeddings
    with open("vector_store.pkl", "wb") as f:
        pickle.dump(vector_store, f)

def load_embeddings():
    """Load saved embeddings"""
    if not os.path.exists("vector_store.pkl"):
        create_embeddings()
    with open("vector_store.pkl", "rb") as f:
        vector_store = pickle.load(f)
    return vector_store
