"""
ingestion/document.py
──────────────────────
Ingests PDF, DOCX, TXT files using LlamaIndex.
Chunks them, attaches source metadata, writes to ChromaDB.
"""
import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter

from backend.config import CHROMA_DB_PATH
from backend.agents.document_agent import extract_document_insights


# ── ChromaDB client (singleton) ───────────────────────────────────────────────
_chroma_client = None
_collection = None


def get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = _chroma_client.get_or_create_collection(
            name="living_resume",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def clear_collection_cache():
    global _chroma_client, _collection
    _collection = None
    _chroma_client = None



def _chunk_id(source: str, chunk_index: int) -> str:
    h = hashlib.md5(f"{source}:{chunk_index}".encode()).hexdigest()[:8]
    return f"{Path(source).stem}_chunk_{chunk_index}_{h}"


def ingest_file(file_path: str, person_name: str = "unknown") -> Dict[str, Any]:
    """
    Ingest a single file. Returns summary dict.
    Steps:
      1. Load with LlamaIndex SimpleDirectoryReader
      2. Sentence-split into ~400 token chunks with 50 token overlap
      3. Store in ChromaDB with rich metadata
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    docs: List[Document] = SimpleDirectoryReader(input_files=[str(path)]).load_data()

    splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(docs)

    collection = get_collection()
    source_label = _infer_source_label(path.name)

    # Perform intelligent multimodal extraction
    insights = extract_document_insights(str(path), person_name)
    
    detected_name = insights.get("candidate_name")
    if detected_name and isinstance(detected_name, str) and detected_name.strip() and detected_name.lower() not in ["unknown", "n/a", "none"]:
        person_name = detected_name.strip()
    else:
        # Fallback: get the very first word of the document (useful if LLM hits rate limits)
        if docs:
            first_text = docs[0].text.strip()
            if first_text:
                first_word = first_text.split()[0]
                first_word = re.sub(r'[^a-zA-Z]', '', first_word)
                if first_word:
                    person_name = first_word.capitalize()
        
    timeline = insights.get("timeline", [])
    keywords = insights.get("keywords", [])

    ids, documents, metadatas = [], [], []
    
    # Inject synthetic chunks for timeline and keywords
    if timeline:
        import json
        timeline_text = f"Master Timeline for {person_name}:\n" + json.dumps(timeline, indent=2)
        ids.append(_chunk_id(path.name, -1))
        documents.append(timeline_text)
        metadatas.append({
            "source": path.name,
            "source_label": source_label,
            "section_index": -1,
            "person": person_name,
            "file_type": "synthetic_timeline",
            "chunk_chars": len(timeline_text),
        })

    if keywords:
        keywords_text = f"Core Competencies and Keywords for {person_name}:\n" + ", ".join(keywords)
        ids.append(_chunk_id(path.name, -2))
        documents.append(keywords_text)
        metadatas.append({
            "source": path.name,
            "source_label": source_label,
            "section_index": -2,
            "person": person_name,
            "file_type": "synthetic_keywords",
            "chunk_chars": len(keywords_text),
        })

    for i, node in enumerate(nodes):
        chunk_id = _chunk_id(path.name, i)
        ids.append(chunk_id)
        documents.append(node.text)
        metadatas.append({
            "source": path.name,
            "source_label": source_label,
            "section_index": i,
            "person": person_name,
            "file_type": path.suffix.lstrip("."),
            "chunk_chars": len(node.text),
        })

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return {
        "file": path.name,
        "source_label": source_label,
        "chunks_ingested": len(ids),
        "total_chars": sum(m["chunk_chars"] for m in metadatas),
        "person_name": person_name,
    }


def ingest_text(
    text: str,
    source_label: str,
    person_name: str = "unknown",
    metadata_extra: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Ingest raw text (e.g. interview transcript, LinkedIn export, crawled page).
    """
    splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)
    doc = Document(text=text, metadata={"source": source_label})
    nodes = splitter.get_nodes_from_documents([doc])

    collection = get_collection()
    ids, documents, metadatas = [], [], []
    for i, node in enumerate(nodes):
        chunk_id = _chunk_id(source_label, i)
        ids.append(chunk_id)
        documents.append(node.text)
        meta = {
            "source": source_label,
            "source_label": source_label,
            "section_index": i,
            "person": person_name,
            "file_type": "text",
            "chunk_chars": len(node.text),
        }
        if metadata_extra:
            meta.update(metadata_extra)
        metadatas.append(meta)

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return {
        "source_label": source_label,
        "chunks_ingested": len(ids),
        "total_chars": sum(m["chunk_chars"] for m in metadatas),
    }


def get_all_chunks() -> List[Dict[str, Any]]:
    """Return all stored chunks as list of {text, metadata}."""
    collection = get_collection()
    result = collection.get(include=["documents", "metadatas"])
    chunks = []
    for doc, meta in zip(result["documents"], result["metadatas"]):
        chunks.append({"text": doc, "metadata": meta})
    return chunks


def query_vector(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Semantic vector search via ChromaDB's built-in embedding.
    Returns list of {text, metadata, score}.
    """
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = max(0.0, 1.0 - dist)
        hits.append({"text": doc, "metadata": meta, "score": score})
    return hits


def collection_stats() -> Dict[str, Any]:
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return {"total_chunks": 0, "sources": []}
    result = collection.get(include=["metadatas"])
    sources = list({m.get("source_label", "unknown") for m in result["metadatas"]})
    return {"total_chunks": count, "sources": sources}


def _infer_source_label(filename: str) -> str:
    name_lower = filename.lower()
    if "resume" in name_lower or "cv" in name_lower:
        return "resume"
    if "linkedin" in name_lower:
        return "linkedin"
    if "transcript" in name_lower or "interview" in name_lower:
        return "interview_transcript"
    if "note" in name_lower:
        return "notes"
    return "document"
