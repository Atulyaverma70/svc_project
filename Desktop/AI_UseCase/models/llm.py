# models/llm.py
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from config import OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY


def get_openai_model():
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-3.5-turbo",
        temperature=0
    )


def get_groq_model():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant"  # Groq’s fastest model
    )


def get_gemini_model():
    return ChatGoogleGenerativeAI(
        api_key=GEMINI_API_KEY,
        model="gemini-1.5-pro"
    )


def get_chat_model(provider="groq"):
    """
    Simple helper: choose provider by string
    """
    provider = provider.lower()
    if provider == "openai":
        return get_openai_model()
    if provider == "gemini":
        return get_gemini_model()
    return get_groq_model()  # default
