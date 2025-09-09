# app.py
import streamlit as st
import os
import sys
import tempfile
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# allow imports from project folders
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# model helpers
from models.llm import get_chat_model
from utils.rag_utils import get_rag_answer
from utils.web_search import web_search
from config import OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY

# Try to import GroqEmbeddings dynamically
try:
    from langchain_groq import GroqEmbeddings
    has_groq = True
except ImportError:
    has_groq = False


def process_uploaded_file(uploaded_file, provider="groq"):
    """
    Process uploaded PDF or TXT file and return a FAISS vectorstore.
    Uses GroqEmbeddings by default if available; falls back to OpenAIEmbeddings.
    """
    from langchain.embeddings.openai import OpenAIEmbeddings

    # Load document
    if uploaded_file.name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    else:
        text = uploaded_file.read().decode("utf-8")
        loader = TextLoader(text)
        docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Select embeddings
    if provider == "groq" and has_groq:
        embeddings = GroqEmbeddings(api_key=GROQ_API_KEY)
    elif provider == "openai":
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    else:
        # default fallback
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_chat_response(chat_model, messages, system_prompt):
    """Get response from the chat model."""
    formatted_messages = [SystemMessage(content=system_prompt)]
    for msg in messages:
        if msg["role"] == "user":
            formatted_messages.append(HumanMessage(content=msg["content"]))
        else:
            formatted_messages.append(AIMessage(content=msg["content"]))
    try:
        response = chat_model.invoke(formatted_messages)
        return response.content
    except Exception as e:
        return f"{type(e).__name__}: {str(e)}"


def instructions_page():
    st.title("Customer Support Chatbot")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")
    st.markdown("""
    ## 🔧 Installation
    ```bash
    pip install -r requirements.txt
    ```
    ## API Key Setup
    Set your API keys in `config/config.py`:
    - `OPENAI_API_KEY`
    - `GEMINI_API_KEY`
    - `GROQ_API_KEY`
    ## How to Use
    1. Go to the **Chat** page.
    2. Upload a PDF/TXT knowledge base if needed.
    3. Start chatting once everything is configured!
    """)


def chat_page():
    st.title("🤖 Customer Support Chatbot")

    provider = st.sidebar.selectbox(
        "Select AI Provider",
        ["Groq (Default)", "OpenAI", "Google Gemini"],
        index=0
    )
    response_mode = st.sidebar.radio("Response Mode:", ["Concise", "Detailed"])

    # keep chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Upload file & build vectorstore
    uploaded_file = st.sidebar.file_uploader(
        "Upload knowledge base file (PDF/TXT/JSON)",
        type=["pdf", "txt", "json"]
    )
    if uploaded_file:
        provider_key = "groq" if provider.startswith("Groq") else provider.lower()
        st.session_state.vectorstore = process_uploaded_file(uploaded_file, provider=provider_key)
        st.sidebar.success(f"Indexed: {uploaded_file.name}")

    # get model
    try:
        if provider.startswith("OpenAI"):
            chat_model = get_chat_model("openai")
        elif provider.startswith("Google Gemini"):
            chat_model = get_chat_model("gemini")
        else:
            chat_model = get_chat_model("groq")  # default
    except Exception as e:
        st.error(str(e))
        return

    # display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # build context: uploaded vectorstore first, else rag_utils/web_search
        context = ""
        if st.session_state.vectorstore:
            retriever = st.session_state.vectorstore.as_retriever()
            retrieved_docs = retriever.get_relevant_documents(prompt)
            context = "\n".join([d.page_content for d in retrieved_docs])
        else:
            context = get_rag_answer(prompt)
            if not context:
                context = web_search(prompt)

        system_prompt = f"""
        You are a helpful customer support assistant for our product.
        Use the following context to answer the user question:
        {context}

        Answer in {'short sentences' if response_mode=='Concise' else 'detailed steps'}.
        """

        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                response = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.set_page_config(
        page_title="Customer Support Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)
        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.session_state.vectorstore = None
                st.rerun()

    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()


if __name__ == "__main__":
    main()
