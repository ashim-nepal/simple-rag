import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, TFAutoModelForQuestionAnswering, AutoTokenizer

# -----------------------------
# Load models (embedding + QA)
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load QA model using TensorFlow backend
qa_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer, framework="tf")

# -----------------------------
# Helper Functions
# -----------------------------
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def build_faiss(chunks):
    vectors = embedder.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, vectors

def get_top_chunks(query, chunks, index, k=3):
    q_vec = embedder.encode([query])
    _, top_indices = index.search(np.array(q_vec), k)
    return [chunks[i] for i in top_indices[0]]

def answer_question_tf(question, context):
    return qa_pipeline(question=question, context=context)["answer"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="📄 PDF Q&A (TensorFlow RAG)", layout="wide")
st.title("📄 PDF Q&A Bot – TensorFlow RAG")

uploaded_pdf = st.file_uploader("📂 Upload your PDF", type="pdf")

if uploaded_pdf:
    with st.spinner("📚 Reading and indexing PDF..."):
        full_text = extract_text(uploaded_pdf)
        chunks = chunk_text(full_text)
        index, _ = build_faiss(chunks)
    st.success("✅ Done! Ask your question:")

    user_question = st.text_input("💬 Ask a question")

    if user_question:
        with st.spinner("🔍 Retrieving answer..."):
            top_chunks = get_top_chunks(user_question, chunks, index)
            context = " ".join(top_chunks)
            answer = answer_question_tf(user_question, context)

        st.markdown("### ✅ Answer")
        st.success(answer)

        with st.expander("📖 Retrieved Context"):
            st.write(context)
