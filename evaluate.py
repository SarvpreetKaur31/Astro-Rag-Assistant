import json
import os
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

index = faiss.read_index("galaxy_index.faiss")
with open("papers_metadata.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

def get_embedding(text):
    result = client.models.embed_content(model="gemini-embedding-001", contents=text)
    return result.embeddings[0].values

def retrieve(question, top_k=3):
    query_emb = np.array([get_embedding(question)], dtype="float32")
    distances, indices = index.search(query_emb, top_k)
    return [papers[i] for i in indices[0]]

# Your test questions
test_questions = [
    "How are spiral and irregular galaxies classified?",
    "What role do galaxy mergers play in morphology?",
    "How does deep learning help classify galaxies?",
    "What are the properties of elliptical galaxies?",
    "How is SDSS used to study galaxy morphology?",
    "What is the Galaxy Zoo project?",
    "How do multiwavelength surveys help classify galaxies?",
    "What is the Kinematic Morphology-Density relation?",
]

all_precisions = []

print("=" * 70)
for q in test_questions:
    print(f"\nQUESTION: {q}")
    retrieved = retrieve(q, top_k=3)
    for i, p in enumerate(retrieved):
        print(f"  [{i+1}] {p['title']}")
        print(f"      {p['abstract']}")
        print(f"      Link: {p['link']}\n")

    print("\n  For each paper above, is it relevant? (y/n)")
    relevant_count = 0
    for i, p in enumerate(retrieved):
        ans = input(f"  Paper [{i+1}] relevant? (y/n): ").strip().lower()
        if ans == "y":
            relevant_count += 1

    precision = relevant_count / len(retrieved)
    all_precisions.append(precision)
    print(f"  --> Precision@3 for this question: {precision:.2f}")
    print("-" * 70)

avg_precision = sum(all_precisions) / len(all_precisions)
print(f"\n\nFINAL AVERAGE PRECISION@3: {avg_precision:.2%}")