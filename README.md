\# 🌌 Astro-RAG: Galaxy Research Assistant



A Retrieval-Augmented Generation (RAG) system that answers questions about galaxy morphology, classification, and mergers — grounded in real astrophysics papers from arXiv.



\## What it does



Ask a question like \*"How are spiral and irregular galaxies classified?"\* and the system:

1\. Retrieves the most relevant research papers using semantic search (FAISS)

2\. Generates a grounded answer using Google's Gemini API, citing only the retrieved papers

3\. Displays source papers ranked by citation count, with a retrieval confidence score



\## Architecture



arXiv API → paper fetching + relevance filtering

↓

Semantic Scholar API → citation-count ranking (proxy for paper quality)

↓

Gemini Embeddings → vector representation of papers

↓

FAISS → similarity search / retrieval

↓

Gemini (gemini-3.5-flash) → grounded answer generation

↓

Streamlit → interactive UI



\## Tech Stack



\- \*\*Python\*\*

\- \*\*Google Gemini API\*\* — embeddings (`gemini-embedding-001`) + generation (`gemini-3.5-flash`)

\- \*\*FAISS\*\* — vector similarity search

\- \*\*arXiv API\*\* — paper retrieval

\- \*\*Semantic Scholar API\*\* — citation-based paper ranking

\- \*\*Streamlit\*\* — interactive UI



\## Key Engineering Decisions



\- \*\*arXiv over Google Scholar\*\*: arXiv provides a free, structured, official API; Google Scholar has no public API and prohibits scraping.

\- \*\*Citation-based ranking\*\*: Used Semantic Scholar's citation counts as a practical proxy for paper quality/impact, since journal-quartile data isn't freely available.

\- \*\*Relevance filtering\*\*: Raw keyword search returned some off-topic results (e.g., generic ML papers); added a keyword-based relevance filter before embedding to improve data quality.

\- \*\*Rate-limit handling\*\*: Implemented exponential backoff for Semantic Scholar API calls to handle free-tier rate limits gracefully.



\## Running Locally



```bash

pip install -r requirements.txt

\# Add your Gemini API key to a .env file:

\# GEMINI\_API\_KEY=your\_key\_here



python fetch\_data.py      # Fetch and rank papers

python build\_index.py     # Generate embeddings, build FAISS index

streamlit run app.py      # Launch the app

```



\## Future Improvements



\- Expand to full paper text (not just abstracts) for deeper retrieval

\- Add topic-balanced sampling (currently pure citation-based ranking can underrepresent less-cited but relevant subtopics like galaxy mergers)

\- Add retrieval evaluation metrics (precision@k)

