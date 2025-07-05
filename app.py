import os
import fitz  # PyMuPDF
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

st.set_page_config(
    page_title="Simple RAG PDF QA | Ashim Nepal",
    page_icon="📄",
    layout="centered"
)


st.title("📄 Retrieval-Augmented Generation (RAG) Question Answering")
st.markdown("""
Welcome to this simple **RAG-based QA app**!  
Upload any PDF document, ask questions about its content, and get precise answers instantly.  
This app uses embeddings and similarity search for retrieval, combined with a question-answering transformer model.  
""")

# Footer function
def footer():
    st.markdown("---")
    st.markdown(
        """
        <p style='text-align: center; font-size: 12px; color: gray;'>
        Developed by **Ashim Nepal** &nbsp; | &nbsp; Portfolio: <a href="https://ashimnepal.com.np" target="_blank">Ashim Nepal</a> &nbsp; | &nbsp; GitHub: <a href="https://github.com/ashim-nepal" target="_blank">GitHub/ashim-nepal</a>
        </p>
        <p style='text-align: center; font-size: 12px; color: gray;'>KRB-05 by Ashim Nepal 07-2025</p>
        """, unsafe_allow_html=True
    )
    
# Upload PDF
pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    doc = fitz.open(stream=pdf.read(), filetype="pdf")
    texts = [page.get_text() for page in doc]
    full_text = " ".join(texts)

    # Split text into chunks
    chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]

    # Embed and build index
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Ask question
    question = st.text_input("Ask a question:")
    if question:
        q_embedding = embedder.encode([question])
        _, I = index.search(q_embedding, k=5)
        context = " ".join([chunks[i] for i in I[0]])

        result = qa_pipeline(question=question, context=context)
        st.write("Answer:", result['answer'])

footer()
