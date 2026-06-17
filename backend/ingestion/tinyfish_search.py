"""
ingestion/tinyfish_search.py
─────────────────────────────
Fetches web footprint data from the Tinyfish Search API.
Formats the results into a text document and ingests it into the knowledge base.
"""
import httpx
from typing import Dict, Any
from urllib.parse import quote_plus

from backend.config import TINYFISH_API_KEY
from backend.ingestion.document import ingest_text

async def fetch_and_ingest_tinyfish(query: str, person_name: str) -> Dict[str, Any]:
    """
    Search the web for a query using Tinyfish API, format the results as a digital footprint,
    and ingest into the RAG vector store.
    """
    if not TINYFISH_API_KEY:
        raise ValueError("TINYFISH_API_KEY is not set. Please add it to your .env file.")

    url = f"https://api.search.tinyfish.ai?query={quote_plus(query)}&location=US&language=en"
    headers = {"X-API-Key": TINYFISH_API_KEY}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    if not results:
        raise ValueError(f"No web results found for query: {query}")

    # Format the results into a markdown-like text
    lines = [f"DIGITAL FOOTPRINT & WEB SEARCH RESULTS FOR: {query}\n"]
    
    for res in results:
        title = res.get("title", "No Title")
        site = res.get("site_name", "Unknown Site")
        url = res.get("url", "")
        snippet = res.get("snippet", "")
        
        lines.append(f"Source: {site} ({url})")
        lines.append(f"Title: {title}")
        lines.append(f"Snippet: {snippet}")
        lines.append("-" * 40)

    combined_text = "\n".join(lines)

    # Ingest the combined text
    ingest_result = ingest_text(
        text=combined_text,
        source_label="tinyfish_web_search",
        person_name=person_name,
        metadata_extra={"query": query}
    )

    return ingest_result
