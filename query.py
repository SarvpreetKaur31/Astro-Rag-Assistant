import json
import os
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load the FAISS index and paper metadata
index = faiss.read_index("galaxy_index.faiss")
with open("papers_metadata.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def retrieve(question, top_k=3):
    query_emb = np.array([get_embedding(question)], dtype="float32")
    distances, indices = index.search(query_emb, top_k)
    results = [papers[i] for i in indices[0]]
    return results

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

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text

# --- Interactive test loop ---
if __name__ == "__main__":
    print("Astro-RAG Assistant (type 'quit' to exit)\n")
    while True:
        question = input("Ask a question: ")
        if question.lower() == "quit":
            break

        retrieved = retrieve(question)
        print("\n--- Retrieved papers ---")
        for p in retrieved:
            print(f"- {p['title']}")

        answer = generate_answer(question, retrieved)
        print(f"\n--- Answer ---\n{answer}\n")