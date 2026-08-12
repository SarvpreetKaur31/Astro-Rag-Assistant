import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import time
import re

queries = [
    "galaxy morphology classification",
    "multiwavelength galaxy survey",
    "spiral irregular galaxy classification",
    "galaxy mergers interacting galaxies",
    "deep learning galaxy morphology",
    "elliptical galaxy structure",
    "galaxy classification convolutional neural network",
    "SDSS galaxy morphology"
]

galaxy_keywords = ["galaxy", "galaxies", "morphology", "spiral", "irregular",
                    "merger", "sdss", "multiwavelength", "elliptical"]

all_papers = []

for query in queries:
    print(f"Fetching papers for: {query}")
    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results=10"
    response = urllib.request.urlopen(url)
    data = response.read().decode("utf-8")
    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        link = entry.find("atom:id", ns).text.strip()
        all_papers.append({"title": title, "abstract": summary, "link": link})
    time.sleep(1)  # be polite to arXiv's API

# Deduplicate
seen = set()
unique_papers = []
for p in all_papers:
    if p["title"] not in seen:
        seen.add(p["title"])
        unique_papers.append(p)

# Filter for relevance
filtered = [p for p in unique_papers
            if any(k in (p["title"] + p["abstract"]).lower() for k in galaxy_keywords)]

print(f"\nRelevant papers found: {len(filtered)}")
print("Fetching citation counts from Semantic Scholar (this ranks quality)...")

def get_citation_count(arxiv_link, retries=4):
    match = re.search(r'(\d{4}\.\d{4,5})', arxiv_link)
    if not match:
        return 0
    arxiv_id = match.group(1)
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=citationCount"

    wait = 5
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("citationCount", 0) or 0
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                print(f"   (rate limited, waiting {wait}s...)")
                time.sleep(wait)
                wait *= 2  # 5s, 10s, 20s, 40s
                continue
            return 0
        except Exception:
            return 0
    return 0
for i, p in enumerate(filtered):
    p["citations"] = get_citation_count(p["link"])
    print(f"[{i+1}/{len(filtered)}] {p['title'][:50]}... -> {p['citations']} citations")
    time.sleep(3)  # Semantic Scholar free tier rate limit

# Sort by citation count, descending (proxy for impact/quality)
filtered.sort(key=lambda x: x["citations"], reverse=True)

# Keep top 25
final_papers = filtered[:25]

with open("galaxy_papers.json", "w", encoding="utf-8") as f:
    json.dump(final_papers, f, indent=2)

print(f"\nSaved top {len(final_papers)} papers, sorted by citation count.")