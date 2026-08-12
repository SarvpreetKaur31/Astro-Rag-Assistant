import json
import os
import numpy as np
import faiss
import streamlit as st
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Astro-RAG Assistant", page_icon="🔭", layout="centered")

# ---- Custom CSS: space theme ----
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0a0e27 0%, #1a1a3e 100%);
    color: #e0e0f0;
}
h1 {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
.paper-card {
    background: rgba(255,255,255,0.05);
    border-left: 3px solid #a78bfa;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.citation-badge {
    display: inline-block;
    background: rgba(167,139,250,0.2);
    color: #c4b5fd;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    margin-left: 8px;
}
.stTextInput input {
    background-color: #1e2140 !important;
    color: #ffffff !important;
    border: 1px solid rgba(167,139,250,0.3) !important;
    border-radius: 8px !important;
}
.stTextInput input::placeholder {
    color: #8888aa !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    index = faiss.read_index("galaxy_index.faiss")
    with open("papers_metadata.json", "r", encoding="utf-8") as f:
        papers = json.load(f)
    return index, papers

index, papers = load_data()

def get_embedding(text):
    result = client.models.embed_content(model="gemini-embedding-001", contents=text)
    return result.embeddings[0].values

def retrieve(question, top_k=5):
    query_emb = np.array([get_embedding(question)], dtype="float32")
    distances, indices = index.search(query_emb, top_k)
    results = [papers[i] for i in indices[0]]
    scores = [float(d) for d in distances[0]]
    return results, scores

def generate_answer(question, retrieved_papers):
    context = "\n\n".join(
        [f"Title: {p['title']}\nAbstract: {p['abstract']}" for p in retrieved_papers]
    )
    prompt = f"""You are an astronomy research assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say so honestly.

Context:
{context}

Question: {question}

Answer:"""
    response = client.models.generate_content(model="gemini-3.5-flash", contents=prompt)
    return response.text

# ---- UI ----
st.title("🌌 Astro-RAG")
st.caption("Ask questions about galaxy morphology & classification — grounded in real, citation-ranked arXiv papers")

# Example questions as clickable buttons
st.write("**Try asking:**")
examples = [
    "How are spiral and irregular galaxies classified?",
    "What role do galaxy mergers play in morphology?",
    "How does deep learning help classify galaxies?"
]
cols = st.columns(3)
clicked_question = None
for i, ex in enumerate(examples):
    if cols[i].button(ex, use_container_width=True):
        clicked_question = ex

question = st.text_input("Or type your own question:", value=clicked_question or "")

if question:
    with st.spinner("🔭 Searching papers..."):
        retrieved, scores = retrieve(question)

    with st.spinner("🧠 Generating answer..."):
        answer = generate_answer(question, retrieved)

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### 📚 Sources")
    for p, score in zip(retrieved, scores):
        confidence = max(0, min(100, int(100 - score * 10)))
        citations = p.get("citations", 0)
        st.markdown(f"""
<div class="paper-card">
<b><a href="{p['link']}" target="_blank" style="color:#a78bfa;">{p['title']}</a></b>
<span class="citation-badge">📈 {citations} citations</span>
<span class="citation-badge">🎯 {confidence}% match</span>
<p style="color:#c0c0d8; font-size:13px; margin-top:6px;">{p['abstract'][:220]}...</p>
</div>
""", unsafe_allow_html=True)