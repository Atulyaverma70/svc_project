# utils/rag_utils.py
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import os

def load_documents():
    docs = []

    # User manual PDF
    manual_path = "knowledge_docs/manuals/user_manual.pdf"
    if os.path.exists(manual_path):
        docs.extend(PyPDFLoader(manual_path).load())

    # FAQs text
    faqs_txt_path = "knowledge_docs/faqs/faqs.txt"
    if os.path.exists(faqs_txt_path):
        docs.extend(TextLoader(faqs_txt_path).load())

    # FAQs JSON
    faqs_json_path = "knowledge_docs/faqs/faq.json"
    if os.path.exists(faqs_json_path):
        loader = JSONLoader(
            faqs_json_path,
            jq_schema=".[] | {question: .question, answer: .answer}",
            text_content=True  # must be True
        )
        for doc in loader.load():
            doc.page_content = f"Q: {doc.metadata['question']}\nA: {doc.metadata['answer']}"
            docs.append(doc)

    # Troubleshooting text
    troubleshooting_path = "knowledge_docs/troubleshooting/troubleshooting.txt"
    if os.path.exists(troubleshooting_path):
        docs.extend(TextLoader(troubleshooting_path).load())

    return docs


def get_vectorstore():
    docs = load_documents()
    if not docs:
        return None
    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY automatically
    return FAISS.from_documents(docs, embeddings)


def get_rag_answer(query):
    try:
        vectorstore = get_vectorstore()
        if not vectorstore:
            return ""
        results = vectorstore.similarity_search(query, k=3)
        return "\n".join([r.page_content for r in results])
    except Exception as e:
        print("RAG search error:", e)
        return ""
