"""
main.py — FastAPI application (Living Resume)
──────────────────────────────────────────────
Routes:
  POST /ingest/file          Upload and ingest a document (PDF, DOCX, TXT)
  POST /ingest/text          Ingest raw text
  POST /ingest/url           Crawl and ingest a website
  POST /ingest/linkedin      Ingest LinkedIn profile (MCP/Proxycurl or PDF)
  POST /ingest/github        Ingest GitHub profile data
  POST /qa                   Ask a question (JSON response)
  GET  /qa/stream            Ask a question (SSE streaming)
  POST /interview/start      Start AI interview session
  POST /interview/answer     Submit answer, get next question
  GET  /graph                Get knowledge graph for visualization
  GET  /graph/communities    Get GraphRAG community summaries
  GET  /stats                System stats (chunks, graph nodes, cache)
  DELETE /reset              Reset all knowledge (dev use)
  GET  /health               Health check
"""
import os
import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.config import UPLOAD_DIR, GEMINI_API_KEY, CHROMA_DB_PATH, GRAPH_DB_PATH
from backend.ingestion.document import ingest_file, ingest_text, collection_stats, get_all_chunks
from backend.ingestion.web_crawler import crawl_url, crawl_site
from backend.agents.crawler_agent import process_crawled_content
from backend.ingestion.linkedin_mcp import fetch_github_data, fetch_linkedin_profile, linkedin_pdf_to_text
from backend.knowledge.graph import get_graph, load_graph_on_startup
from backend.knowledge.bm25_index import rebuild_bm25_from_chroma, get_bm25_index
from backend.cache.kv_cache import get_cache_stats, estimate_token_savings
from backend.agents.qa_agent import ask_question, stream_answer, get_profile_summary, get_suggested_questions
from backend.agents.interview_agent import start_or_continue_interview
from backend.agents.graph_agent import extract_and_populate_graph


# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize knowledge graph and BM25 index on startup."""
    load_graph_on_startup()
    count = rebuild_bm25_from_chroma()
    print(f"[Startup] BM25 built with {count} chunks.")
    if not GEMINI_API_KEY:
        print("[WARNING] GEMINI_API_KEY not set. Set it in .env file.")
    yield
    # Shutdown: save graph
    get_graph().save()
    print("[Shutdown] Graph saved.")


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Living Resume API",
    version="1.0.0",
    description="Conversational knowledge base for a person's professional history.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    text: str
    source_label: str = "notes"
    person_name: str = "unknown"
    reset_kb: bool = False

class IngestURLRequest(BaseModel):
    url: str
    person_name: str = "unknown"
    crawl_site: bool = False
    max_pages: int = 5
    reset_kb: bool = False

class LinkedInRequest(BaseModel):
    linkedin_url: str
    person_name: str = "unknown"
    reset_kb: bool = False

class GitHubRequest(BaseModel):
    username: str
    person_name: str = "unknown"
    reset_kb: bool = False

class QARequest(BaseModel):
    query: str
    person_name: str = "unknown"
    session_id: str = "default"

class InterviewStartRequest(BaseModel):
    person_name: str
    session_id: str = "default"

class InterviewAnswerRequest(BaseModel):
    person_name: str
    session_id: str = "default"
    answer: str


# ── Shared reset helper ──────────────────────────────────────────────────────

def reset_all():
    """Wipe ChromaDB, graph, and BM25 cache so every ingest starts fresh."""
    import chromadb
    import networkx as nx

    # 1. Clear ChromaDB vector store
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        client.delete_collection("living_resume")
        from backend.ingestion.document import clear_collection_cache
        clear_collection_cache()
        print("[Reset] ChromaDB cleared.")
    except Exception as e:
        print(f"[Reset] ChromaDB clear skipped: {e}")

    # 2. Clear knowledge graph
    try:
        graph = get_graph()
        graph.g = nx.DiGraph()
        graph._community_map = {}
        graph._community_summaries = {}
        graph.save()
        print("[Reset] Graph cleared.")
    except Exception as e:
        print(f"[Reset] Graph clear skipped: {e}")

    # 3. Rebuild empty BM25 index
    try:
        rebuild_bm25_from_chroma()
        print("[Reset] BM25 index reset.")
    except Exception as e:
        print(f"[Reset] BM25 reset skipped: {e}")


# ── Ingestion routes ──────────────────────────────────────────────────────────

@app.post("/ingest/file")
async def ingest_file_route(
    file: UploadFile = File(...),
    person_name: str = Form("unknown"),
    reset_kb: bool = Form(False),
):
    """Upload and ingest a PDF, DOCX, or TXT file."""
    if reset_kb:
        reset_all()

    save_path = Path(UPLOAD_DIR) / file.filename
    content = await file.read()
    save_path.write_bytes(content)

    try:
        result = ingest_file(str(save_path), person_name=person_name)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    # Check if this is a LinkedIn PDF export
    if "linkedin" in file.filename.lower():
        # Try specialized LinkedIn PDF parser
        linkedin_text = linkedin_pdf_to_text(str(save_path))
        if linkedin_text and len(linkedin_text) > 100:
            result = ingest_text(linkedin_text, "linkedin", person_name)

    actual_name = result.get("person_name", person_name)

    chunks = get_all_chunks()
    new_chunks = chunks[-result["chunks_ingested"]:] if result["chunks_ingested"] else []
    extraction = await extract_and_populate_graph(new_chunks, actual_name) if new_chunks else {}

    return {
        "status": "success", 
        "ingestion": result, 
        "extraction": extraction,
        "person_name": actual_name
    }


@app.post("/ingest/text")
async def ingest_text_route(req: IngestTextRequest):
    """Ingest raw text directly."""
    if req.reset_kb:
        reset_all()
    result = ingest_text(req.text, req.source_label, req.person_name)
    chunks = get_all_chunks()
    new_chunks = chunks[-result["chunks_ingested"]:] if result["chunks_ingested"] else []
    extraction = await extract_and_populate_graph(new_chunks, req.person_name) if new_chunks else {}
    return {"status": "success", "ingestion": result, "extraction": extraction}


@app.post("/ingest/url")
async def ingest_url_route(req: IngestURLRequest):
    """Crawl a URL or entire site and ingest content."""
    if req.reset_kb:
        reset_all()
    if req.crawl_site:
        pages = await crawl_site(req.url, max_pages=req.max_pages)
    else:
        pages = [await crawl_url(req.url)]

    ingested = []
    for page in pages:
        if page.get("success") and page.get("markdown"):
            import json
            insights = process_crawled_content(page["url"], page["markdown"])
            
            enhanced_text = f"Summary for {page['url']}:\n{insights.get('summary', '')}\n\n"
            
            if insights.get("timeline"):
                enhanced_text += f"Timeline Extracted:\n{json.dumps(insights['timeline'], indent=2)}\n\n"
            if insights.get("keywords"):
                enhanced_text += f"Keywords/Skills:\n{', '.join(insights['keywords'])}\n\n"
                
            # Append original markdown so we don't lose exact quotes
            enhanced_text += f"--- RAW CONTENT ---\n{page['markdown']}"
            
            result = ingest_text(
                enhanced_text,
                source_label=f"website:{page['url'][:50]}",
                person_name=req.person_name,
                metadata_extra={"url": page["url"], "title": page.get("title", "")},
            )
            ingested.append(result)

    total_new = sum(r["chunks_ingested"] for r in ingested)
    chunks = get_all_chunks()
    new_chunks = chunks[-total_new:] if total_new else []
    extraction = await extract_and_populate_graph(new_chunks, req.person_name) if new_chunks else {}

    return {
        "status": "success",
        "pages_crawled": len(pages),
        "total_chunks_ingested": total_new,
        "extraction": extraction,
        "errors": [p["error"] for p in pages if not p.get("success")]
    }


class IngestTinyfishRequest(BaseModel):
    query: str
    person_name: str = "unknown"
    reset_kb: bool = False

@app.post("/ingest/tinyfish")
async def ingest_tinyfish_route(req: IngestTinyfishRequest):
    """Search the web via Tinyfish API and ingest the digital footprint."""
    if req.reset_kb:
        reset_all()
    from backend.ingestion.tinyfish_search import fetch_and_ingest_tinyfish
    try:
        result = await fetch_and_ingest_tinyfish(req.query, req.person_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    chunks = get_all_chunks()
    new_chunks = chunks[-result["chunks_ingested"]:] if result["chunks_ingested"] else []
    extraction = await extract_and_populate_graph(new_chunks, req.person_name) if new_chunks else {}
    
    return {"status": "success", "ingestion": result, "extraction": extraction}


@app.post("/ingest/linkedin")
async def ingest_linkedin_route(req: LinkedInRequest):
    """
    Fetch LinkedIn profile via FastMCP/Proxycurl and ingest.
    Falls back gracefully if PROXYCURL_API_KEY not set.
    """
    if req.reset_kb:
        reset_all()
    data = await fetch_linkedin_profile(req.linkedin_url)

    if not data.get("available"):
        # Fallback: use Tinyfish to search for the LinkedIn URL
        from backend.ingestion.tinyfish_search import fetch_and_ingest_tinyfish
        try:
            result = await fetch_and_ingest_tinyfish(req.linkedin_url, req.person_name)
            chunks = get_all_chunks()
            new_chunks = chunks[-result["chunks_ingested"]:] if result["chunks_ingested"] else []
            extraction = await extract_and_populate_graph(new_chunks, req.person_name) if new_chunks else {}
            return {"status": "success", "ingestion": result, "extraction": extraction, "note": "Used Tinyfish fallback"}
        except Exception as e:
            return {
                "status": "fallback",
                "message": data.get("message", "LinkedIn MCP unavailable") + f" AND Tinyfish fallback failed: {e}",
                "suggestion": "Upload your LinkedIn PDF export via /ingest/file instead.",
            }

    result = ingest_text(
        data["text"],
        source_label="linkedin",
        person_name=req.person_name,
        metadata_extra={"linkedin_url": req.linkedin_url},
    )
    chunks = get_all_chunks()
    new_chunks = chunks[-result["chunks_ingested"]:] if result["chunks_ingested"] else []
    extraction = await extract_and_populate_graph(new_chunks, req.person_name) if new_chunks else {}

    return {
        "status": "success",
        "source": data.get("source", "linkedin"),
        "ingestion": result,
        "extraction": extraction,
    }


@app.post("/ingest/github")
async def ingest_github_route(req: GitHubRequest):
    """Fetch GitHub profile and ingest repos."""
    if req.reset_kb:
        reset_all()
    data = await fetch_github_data(req.username)
    if not data.get("available"):
        raise HTTPException(status_code=400, detail=data.get("error", "GitHub fetch failed"))

    result = ingest_text(
        data["text"],
        source_label="github",
        person_name=req.person_name,
        metadata_extra={"github_username": req.username},
    )
    chunks = get_all_chunks()
    new_chunks = chunks[-result["chunks_ingested"]:] if result["chunks_ingested"] else []
    extraction = await extract_and_populate_graph(new_chunks, req.person_name) if new_chunks else {}

    return {
        "status": "success",
        "repos_found": len(data.get("repos", [])),
        "ingestion": result,
        "extraction": extraction,
    }


# ── QA routes ─────────────────────────────────────────────────────────────────

@app.post("/qa")
async def qa_route(req: QARequest):
    """Ask a question, get a JSON response."""
    result = await ask_question(req.query, req.person_name, req.session_id)
    return result


@app.get("/qa/stream")
async def qa_stream_route(
    query: str,
    person_name: str = "unknown",
    session_id: str = "default",
):
    """
    Ask a question with SSE streaming response.
    Each event is a JSON line: {"type": "token"|"retrieval"|"done", ...}
    """
    async def event_generator():
        async for chunk in stream_answer(query, person_name, session_id):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Interview routes ───────────────────────────────────────────────────────────

@app.post("/interview/start")
async def interview_start_route(req: InterviewStartRequest):
    """Start a new interview session. Returns first question + pre-coverage info."""
    result = await start_or_continue_interview(req.person_name, req.session_id)
    return result


@app.post("/interview/answer")
async def interview_answer_route(req: InterviewAnswerRequest):
    """Submit an answer. Returns the next question, extracted entities, and progress."""
    result = await start_or_continue_interview(
        req.person_name,
        req.session_id,
        human_answer=req.answer,
    )
    return result


@app.post("/interview/reset")
async def interview_reset_route(req: InterviewStartRequest):
    """Clear session state so the interview can be restarted fresh."""
    from backend.agents.interview_agent import _sessions
    key = f"{req.person_name}_{req.session_id}"
    _sessions.pop(key, None)
    return {"ok": True, "message": "Interview session cleared."}


# ── Graph + Stats routes ──────────────────────────────────────────────────────

@app.get("/graph")
async def graph_route():
    """Get knowledge graph data for frontend visualization."""
    graph = get_graph()
    return {
        "graph": graph.to_frontend_json(),
        "stats": graph.stats,
        "global_summary": graph.global_summary(),
    }


@app.get("/graph/communities")
async def graph_communities_route():
    """Get GraphRAG community summaries (for broad career-arc questions)."""
    graph = get_graph()
    summaries = graph.community_summaries()
    return {
        "communities": summaries,
        "total_communities": len(summaries),
    }


@app.get("/profile/summary")
async def profile_summary_route(person_name: str = "unknown"):
    """Get a structured profile summary from the knowledge graph."""
    return get_profile_summary(person_name)


@app.get("/profile")
async def profile_route(person_name: str = "unknown"):
    """Alias for /profile/summary — used by the biography chat UI."""
    return get_profile_summary(person_name)


@app.get("/chat/suggestions")
async def chat_suggestions_route(person_name: str = "unknown"):
    """Get suggested chat questions based on graph contents."""
    return {"suggestions": get_suggested_questions(person_name)}



@app.get("/stats")
async def stats_route():
    """System-wide stats."""
    chroma_stats = collection_stats()
    graph = get_graph()
    cache_stats = get_cache_stats().to_dict()
    bm25_size = get_bm25_index().size

    # Estimate token savings based on current bio size
    chunks = get_all_chunks()
    bio_text = "\n\n".join(c["text"] for c in chunks[:30])[:6000]
    savings_estimate = estimate_token_savings(bio_text)

    return {
        "vector_store": chroma_stats,
        "graph": graph.stats,
        "bm25": {"index_size": bm25_size},
        "cache": cache_stats,
        "cache_estimate": savings_estimate,
    }


@app.delete("/reset")
async def reset_route():
    """Reset all knowledge (for development/testing)."""
    import chromadb

    # Clear ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        client.delete_collection("living_resume")
    except Exception:
        pass

    # Clear graph
    import networkx as nx
    graph = get_graph()
    graph.g = nx.DiGraph()
    graph._community_map = {}
    graph._community_summaries = {}
    graph.save()

    # Rebuild BM25 (will be empty)
    rebuild_bm25_from_chroma()

    return {"status": "reset complete"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "api_key_set": bool(GEMINI_API_KEY),
        "model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        "chunks": collection_stats().get("total_chunks", 0),
        "graph_nodes": get_graph().stats.get("total_nodes", 0),
    }
