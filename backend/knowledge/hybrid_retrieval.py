"""
knowledge/hybrid_retrieval.py
──────────────────────────────
Reciprocal Rank Fusion (RRF) combines:
  1. Dense vector search (ChromaDB semantic similarity)
  2. Sparse BM25 (keyword/exact-match)
  3. Graph neighbourhood (structural context — local)
  4. GraphRAG community context (global career-arc summaries)

RRF formula: score(d) = Σ 1 / (k + rank_i(d))
  k=60 is the standard constant that dampens high-rank outliers.

Returns a fused, reranked list with confidence score and source citations.
"""
from typing import List, Dict, Any

from backend.config import (
    TOP_K_VECTOR, TOP_K_BM25, FINAL_TOP_K, CONFIDENCE_THRESHOLD
)

RRF_K = 60  # Standard RRF constant


def hybrid_retrieve(
    query: str,
    person_name: str = "unknown",
    strategy: str = "hybrid",   # "kg_first" | "vector_first" | "hybrid"
) -> Dict[str, Any]:
    """
    Main retrieval entry point.
    strategy controls weighting:
      - "kg_first"     → boost graph context, use fewer vector chunks
      - "vector_first" → boost vector chunks, skip community context
      - "hybrid"       → equal weighting (default)
    Returns:
      {
        "chunks": [...],            # top FINAL_TOP_K fused chunks
        "confidence": 0.0-1.0,
        "graph_context": "...",     # local graph neighbourhood text
        "community_context": "...", # GraphRAG community summaries
        "sources_used": [...],      # which retrieval paths fired
        "should_refuse": bool,
        "query_entities": [...],
      }
    """
    from backend.ingestion.document import query_vector
    from backend.knowledge.bm25_index import get_bm25_index
    from backend.knowledge.graph import get_graph

    sources_used = []

    # How many vector chunks to pull based on strategy
    vec_k = TOP_K_VECTOR if strategy != "kg_first" else max(2, TOP_K_VECTOR // 2)

    # ── 1. Vector retrieval ───────────────────────────────────────────────────
    vector_hits = query_vector(query, top_k=vec_k)
    if vector_hits:
        sources_used.append("vector")

    # ── 2. BM25 retrieval ─────────────────────────────────────────────────────
    bm25_index = get_bm25_index()
    bm25_hits  = bm25_index.query(query, top_k=TOP_K_BM25)
    if bm25_hits:
        sources_used.append("bm25")

    # ── 3. Graph local retrieval ──────────────────────────────────────────────
    graph          = get_graph()
    query_entities = _extract_query_entities(query, graph)
    graph_context  = ""
    if query_entities:
        hops = 3 if strategy == "kg_first" else 2
        graph_context = graph.neighbours_as_text(query_entities, hops=hops)
        if graph_context:
            sources_used.append("graph_local")

    # ── 4. GraphRAG community context ─────────────────────────────────────────
    community_context = ""
    if strategy != "vector_first":
        if query_entities:
            community_context = graph.community_context_for_query(query_entities)
            if community_context:
                sources_used.append("graph_community")
        if not query_entities and _is_broad_query(query):
            community_context = graph.global_summary()
            if community_context:
                sources_used.append("graph_global")

    # ── 5. RRF Fusion ─────────────────────────────────────────────────────────
    fused      = _rrf_fuse(vector_hits, bm25_hits)
    top_chunks = fused[:FINAL_TOP_K]

    # ── 6. Confidence scoring ─────────────────────────────────────────────────
    has_graph  = bool(graph_context or community_context)
    confidence = _compute_confidence(top_chunks, vector_hits, bm25_hits, has_graph)

    # For KG-first: if graph found strong context, lower the refusal bar
    kg_bonus     = 0.15 if (strategy == "kg_first" and has_graph) else 0.0
    should_refuse = (confidence + kg_bonus) < CONFIDENCE_THRESHOLD and not has_graph

    return {
        "chunks":            top_chunks,
        "confidence":        confidence,
        "graph_context":     graph_context,
        "community_context": community_context,
        "sources_used":      sources_used,
        "should_refuse":     should_refuse,
        "query_entities":    query_entities,
    }


def _rrf_fuse(
    vector_hits: List[Dict[str, Any]],
    bm25_hits: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion across vector and BM25 results.
    Deduplicates by text prefix, returns merged list sorted by RRF score.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    def _text_key(chunk: Dict) -> str:
        return chunk["text"][:100].strip()

    for rank, hit in enumerate(vector_hits):
        key = _text_key(hit)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = {**hit, "retrieval_sources": ["vector"]}
        elif "vector" not in chunk_map[key]["retrieval_sources"]:
            chunk_map[key]["retrieval_sources"].append("vector")

    for rank, hit in enumerate(bm25_hits):
        key = _text_key(hit)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = {**hit, "retrieval_sources": ["bm25"]}
        elif "bm25" not in chunk_map[key]["retrieval_sources"]:
            chunk_map[key]["retrieval_sources"].append("bm25")

    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
    result = []
    for key in sorted_keys:
        chunk = chunk_map[key]
        chunk["rrf_score"] = rrf_scores[key]
        result.append(chunk)

    return result


def _compute_confidence(
    top_chunks: List[Dict],
    vector_hits: List[Dict],
    bm25_hits: List[Dict],
    has_graph_context: bool,
) -> float:
    """
    Heuristic confidence score (0.0 – 1.0).
    Higher when:
      - Top vector score is high
      - Multiple retrieval paths agree (chunk appears in both vector + BM25)
      - Graph context exists (local or community)
    """
    if not top_chunks and not has_graph_context:
        return 0.0

    base = vector_hits[0]["score"] if vector_hits else 0.0

    # Boost for cross-path agreement
    multi_source = sum(
        1 for c in top_chunks
        if len(c.get("retrieval_sources", [])) > 1
    )
    agreement_boost = min(0.15, multi_source * 0.05)

    # Boost for graph context
    graph_boost = 0.1 if has_graph_context else 0.0

    confidence = min(1.0, base + agreement_boost + graph_boost)
    return round(confidence, 3)


def _extract_query_entities(query: str, graph) -> List[str]:
    """
    Simple entity extraction from query: check which graph node names appear in query.
    """
    q_lower = query.lower()
    matches = []
    for _, data in graph.g.nodes(data=True):
        name = data.get("name", "")
        if name and len(name) > 2 and name.lower() in q_lower:
            matches.append(name)
    return matches[:5]


def _is_broad_query(query: str) -> bool:
    """Detect broad career-arc questions that benefit from global graph summary."""
    broad_terms = [
        "overall", "background", "summary", "experience", "career",
        "tell me about", "who is", "what has", "describe",
    ]
    q_lower = query.lower()
    return any(term in q_lower for term in broad_terms)


def format_chunks_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a clean context block for the LLM."""
    if not chunks:
        return "[No relevant information found in knowledge base]"
    lines = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source_label", meta.get("source", "document"))
        section = meta.get("section_index", "")
        label = f"{source} §{section}" if section != "" else source
        rrf = chunk.get("rrf_score", chunk.get("score", 0))
        paths = "/".join(chunk.get("retrieval_sources", []))
        lines.append(f"[Source: {label} | Score: {rrf:.3f} | via: {paths}]")
        lines.append(chunk["text"].strip())
        lines.append("")
    return "\n".join(lines)
