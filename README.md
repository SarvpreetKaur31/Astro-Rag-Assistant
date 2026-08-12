# Astro-RAG: Galaxy Research Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions about galaxy morphology, classification, and mergers, grounded in real astrophysics papers from arXiv.

## What it does

Ask a question like "How are spiral and irregular galaxies classified?" and the system:

1. Retrieves the most relevant research papers using semantic search (FAISS)
2. Generates a grounded answer using Google's Gemini API, citing only the retrieved papers
3. Displays source papers ranked by citation count, with a retrieval confidence score

## Architecture

arXiv API fetches and filters papers for relevance, then Semantic Scholar API ranks them by citation count, then Gemini Embeddings vectorizes the papers, then FAISS performs similarity search, then Gemini (gemini-3.5-flash) generates grounded answers, all displayed through a Streamlit UI.

## Tech Stack

- Python
- Google Gemini API - embeddings (gemini-embedding-001) and generation (gemini-3.5-flash)
- FAISS - vector similarity search
- arXiv API - paper retrieval
- Semantic Scholar API - citation-based paper ranking
- Streamlit - interactive UI

## Key Engineering Decisions

- arXiv over Google Scholar: arXiv provides a free, structured, official API; Google Scholar has no public API and prohibits scraping.
- Citation-based ranking: Used Semantic Scholar's citation counts as a practical proxy for paper quality and impact, since journal-quartile data isn't freely available.
- Relevance filtering: Raw keyword search returned some off-topic results (e.g., generic ML papers); added a keyword-based relevance filter before embedding to improve data quality.
- Rate-limit handling: Implemented exponential backoff for Semantic Scholar API calls to handle free-tier rate limits gracefully.

## Running Locally

Install dependencies with pip install -r requirements.txt, add your Gemini API key to a .env file as GEMINI_API_KEY=your_key_here, then run python fetch_data.py to fetch and rank papers, python build_index.py to generate embeddings and build the FAISS index, and streamlit run app.py to launch the app.

## Future Improvements

- Expand to full paper text (not just abstracts) for deeper retrieval
- Add topic-balanced sampling, since pure citation-based ranking can underrepresent less-cited but relevant subtopics like galaxy mergers
- Add retrieval evaluation metrics (precision@k)
