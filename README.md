# Living-Resume

An AI-powered digital biography that transforms static resumes and LinkedIn exports into an interactive, multi-modal Knowledge Graph. Features a conversational voice UI, automated timeline extraction, and real-time document intelligence powered by Gemini.

## Architecture

```
Ingestion Layer
  PDF/DOCX → LlamaIndex   |  LinkedIn → FastMCP (Proxycurl)
  Website → Crawl4AI       |  Live Interview → LangGraph

Knowledge Layer  
  ChromaDB (vector)  +  BM25 (sparse)  +  NetworkX (graph)
  GraphRAG community detection          |  RRF fusion

Agent Layer (LangGraph)
  Router → QA / Graph / Interview
```

## Setup

1. Copy `.env.example` to `.env` and add your API keys.
2. Setup Backend:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```
3. Setup Frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
