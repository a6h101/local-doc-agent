import streamlit as st
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import pypdf
import tempfile
import os

st.title("Local Document Q&A Agent")
st.caption("Powered by Llama 3.2 running locally via Ollama")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
chroma_client = chromadb.Client()

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    with st.spinner("Reading and indexing document..."):
        # Extract text from PDF
        reader = pypdf.PdfReader(tmp_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()

        # Split into chunks
        chunk_size = 500
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

        # Store in chromadb
        collection = chroma_client.get_or_create_collection("docs")
        embeddings = embedder.encode(chunks).tolist()
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[str(i) for i in range(len(chunks))]
        )

    st.success(f"Document indexed. {len(chunks)} chunks created.")

    question = st.text_input("Ask something about the document")

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            # Find relevant chunks
            query_embedding = embedder.encode([question]).tolist()
            results = collection.query(query_embeddings=query_embedding, n_results=3)
            context = "\n".join(results["documents"][0])

            # Ask Ollama
            response = ollama.chat(
                model="llama3.2",
                messages=[{
                    "role": "user",
                    "content": f"""Answer based only on this context:
{context}

Question: {question}
If the answer isn't in the context, say 'I don't know'."""
                }]
            )

            st.write("**Answer:**")
            st.write(response["message"]["content"])

    os.unlink(tmp_path)