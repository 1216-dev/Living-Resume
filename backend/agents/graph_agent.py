import json
import logging
import os
import re
from typing import List, Dict, Any, TypedDict, Annotated
import operator

from google.genai import types
from langgraph.graph import StateGraph, END

from backend.config import GEMINI_MODEL, GEMINI_API_KEY
from backend.knowledge.graph import get_graph
from backend.knowledge.bm25_index import rebuild_bm25_from_chroma
from backend.agents.gemini_client import gemini_generate

logger = logging.getLogger(__name__)


BATCH_SIZE = 5

class GraphState(TypedDict):
    chunks: List[Dict[str, Any]]
    person_name: str
    current_index: int
    raw_entities: Annotated[List[Dict[str, Any]], operator.add]
    raw_relationships: Annotated[List[Dict[str, Any]], operator.add]
    resolved_entities: List[Dict[str, Any]]
    resolved_relationships: List[Dict[str, Any]]


def extract_node(state: GraphState) -> Dict[str, Any]:
    """Extracts entities and relationships from the current batch of chunks."""
    idx = state["current_index"]
    batch = state["chunks"][idx : idx + BATCH_SIZE]
    person_name = state["person_name"]
    
    batch_text = "\n\n---\n\n".join(
        f"[Source: {c['metadata'].get('source_label', 'doc')} §{c['metadata'].get('section_index', 0)}]\n{c['text']}"
        for c in batch
    )
    
    prompt = f"""Extract all professional entities and relationships from these text chunks about {person_name}.

TEXT:
{batch_text[:4000]}

You are a deep-extraction Knowledge Graph agent. You must aggressively extract EVERY single entity and relationship, going far beyond explicit keywords.

REQUIRED HIERARCHY & ONTOLOGY:
1. For every Company/Organization, extract:
   - ROLE (e.g., Software Engineer)
   - PROJECTS (e.g., ETL Pipeline Migration)
   - ACHIEVEMENTS (e.g., Reduced latency by 40%)
   - RESPONSIBILITIES (e.g., Managed cloud infrastructure)
2. For every Project (including resume bullet points which MUST be treated as Projects), extract:
   - SKILLS (e.g., Machine Learning, SQL)
   - TECHNOLOGIES/FRAMEWORKS/LIBRARIES/DATABASES/CLOUD PLATFORMS (e.g., Python, Kafka, Spark, PostgreSQL, Docker, AWS)
   - TOOLS (e.g., Git, Jenkins)
   - ACHIEVEMENTS (e.g., Processed 10M+ records/day)
   - METHODOLOGIES / RESEARCH AREAS

CRITICAL INSTRUCTIONS:
- DEEP EXTRACTION: If a project mentions "distributed ETL pipeline processing satellite data", you MUST infer and extract the underlying technologies (e.g., Python, Apache Spark, Kafka, AWS) if evidence exists anywhere in the text.
- Do not stop at explicitly listed skills. Be absolutely exhaustive. Extract every single framework, library, tool, and methodology.
- Link Skills, Tech, Tools, and Achievements DIRECTLY to the PROJECT they were used in, NOT loosely to the Company.
- Link the PROJECT to the COMPANY.

OUTPUT FORMAT:
  "entities": [
    {{"type": "PERSON|COMPANY|ROLE|SKILL|PROJECT|DEGREE|LOCATION|TOOL|TECHNOLOGY|FRAMEWORK|DATASET|CERTIFICATION|PUBLICATION|ACHIEVEMENT|RESPONSIBILITY", "name": "exact name", "context": "brief description"}}
  ],
  "relationships": [
    {{"from": "entity_name", "relation": "WORKED_AS|BUILT_PROJECT|ACHIEVED|RESPONSIBLE_FOR|USED_SKILL|USED_TECH|USED_TOOL|USED_FRAMEWORK|USED_DATASET|STUDIED_AT|PUBLISHED|LOCATED_IN", "to": "entity_name"}}
  ]
}}
"""
    try:
        raw = gemini_generate(
            prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        
        return {
            "raw_entities": data.get("entities", []),
            "raw_relationships": data.get("relationships", []),
            "current_index": idx + BATCH_SIZE
        }
    except Exception as e:
        logger.error("[GraphAgent] Extraction failed on batch %d: %s", idx, e)
        return {"current_index": idx + BATCH_SIZE}


def resolve_node(state: GraphState) -> Dict[str, Any]:
    """Merges and deduplicates raw entities and relationships."""
    raw_e = state.get("raw_entities", [])
    raw_r = state.get("raw_relationships", [])
    
    if not raw_e:
        return {"resolved_entities": [], "resolved_relationships": []}
    
    prompt = f"""You are a strict Knowledge Graph Normalization agent.
Below is a list of raw extracted entities and relationships.

CRITICAL INSTRUCTIONS:
1. Entity Normalization: Merge duplicate entities aggressively. Handle casing (PyTorch == PYTORCH == pytorch -> PyTorch). Resolve acronyms (ML == Machine Learning -> Machine Learning).
2. Standardize Relationships: Ensure relationships strictly use the uppercase snake_case format provided in the schema (e.g., WORKED_AT, USED).
3. Do not drop any valid entities. The graph must be completely exhaustive.

Raw Entities:
{json.dumps(raw_e[:200])}

Raw Relationships:
{json.dumps(raw_r[:200])}

Return ONLY JSON matching this format:
{{
  "entities": [...],
  "relationships": [...]
}}
"""
    try:
        raw = gemini_generate(
            prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        return {
            "resolved_entities": data.get("entities", []),
            "resolved_relationships": data.get("relationships", [])
        }
    except Exception as e:
        logger.error("[GraphAgent] Resolution failed: %s", e)
        return {"resolved_entities": raw_e, "resolved_relationships": raw_r}


def commit_node(state: GraphState) -> Dict[str, Any]:
    """Writes the resolved entities to the graph and triggers community detection."""
    graph = get_graph()
    
    data = {
        "entities": state.get("resolved_entities", []),
        "relationships": state.get("resolved_relationships", [])
    }
    
    if state["chunks"]:
        source = state["chunks"][0]["metadata"].get("source_label", "document")
        graph.add_entities_from_extraction(data, source=source)
        
    graph.compute_communities()
    graph.save()
    rebuild_bm25_from_chroma()
    return {}


def should_continue(state: GraphState) -> str:
    if state["current_index"] < len(state["chunks"]):
        return "extract"
    return "resolve"


def build_graph_agent() -> StateGraph:
    workflow = StateGraph(GraphState)
    
    workflow.add_node("extract", extract_node)
    workflow.add_node("resolve", resolve_node)
    workflow.add_node("commit", commit_node)
    
    workflow.set_entry_point("extract")
    workflow.add_conditional_edges("extract", should_continue, {
        "extract": "extract",
        "resolve": "resolve"
    })
    workflow.add_edge("resolve", "commit")
    workflow.add_edge("commit", END)
    
    return workflow.compile()


async def extract_and_populate_graph(chunks: List[Dict[str, Any]], person_name: str) -> Dict[str, Any]:
    """Entry point for the new LangGraph-based extraction pipeline."""
    if not chunks:
        return {"entities_added": 0, "relationships_added": 0}
        
    agent = build_graph_agent()
    
    initial_state = {
        "chunks": chunks,
        "person_name": person_name,
        "current_index": 0,
        "raw_entities": [],
        "raw_relationships": [],
        "resolved_entities": [],
        "resolved_relationships": []
    }
    
    # Run the graph synchronously for now
    final_state = agent.invoke(initial_state)
    
    graph = get_graph()
    
    return {
        "entities_added": len(final_state.get("resolved_entities", [])),
        "relationships_added": len(final_state.get("resolved_relationships", [])),
        "graph_nodes": graph.stats["total_nodes"],
        "graph_edges": graph.stats["total_edges"],
        "graph_communities": graph.stats.get("communities", 0),
        "bm25_chunks": len(get_all_chunks()) if 'get_all_chunks' in globals() else 0
    }

# Helper just to get bm25_chunks properly
def get_all_chunks():
    from backend.ingestion.document import get_all_chunks as _get
    return _get()
