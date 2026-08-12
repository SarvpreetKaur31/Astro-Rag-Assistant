import json
import os
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load your papers
with open("galaxy_papers.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

print(f"Loaded {len(papers)} papers. Generating embeddings...")

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

embeddings = []
for i, paper in enumerate(papers):
    text = paper["title"] + ". " + paper["abstract"]
    emb = get_embedding(text)
    embeddings.append(emb)
    print(f"[{i+1}/{len(papers)}] Embedded: {paper['title'][:60]}...")

embeddings_array = np.array(embeddings, dtype="float32")

dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

faiss.write_index(index, "galaxy_index.faiss")
with open("papers_metadata.json", "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2)

print(f"\nDone! Indexed {len(papers)} papers into FAISS.")