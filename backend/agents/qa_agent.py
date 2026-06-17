"""
agents/qa_agent.py  —  Biographer Agent Layer
──────────────────────────────────────────────
LangGraph agent pipeline:

  User Query
      ↓
  [intent_node]        — classify intent category + retrieval strategy
      ↓
  [retrieve_node]      — KG-first (factual) or vector-first (narrative) or hybrid
      ↓
  [answer_node]        — Gemini biographer response (grounded, structured)
      ↓
  [memo_node]          — persist confirmed facts across sessions (SqliteSaver)

Response format:
  {
    answer:     str,
    key_facts:  [str, ...],
    sources:    [str, ...],
    confidence: float,
    intent:     str,
    should_refuse: bool,
  }
"""
import json
import logging
import os
import re
from typing import AsyncGenerator, Dict, Any, List, Optional, TypedDict, Annotated
import operator

from google.genai import types
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from backend.config import GEMINI_MODEL, SQLITE_PATH, CONFIDENCE_THRESHOLD
from backend.knowledge.hybrid_retrieval import hybrid_retrieve, format_chunks_for_prompt
from backend.cache.kv_cache import build_system_prompt_with_cache, record_usage, get_cache_stats
from backend.knowledge.graph import get_graph
from backend.agents.gemini_client import (
    gemini_generate, gemini_stream, GeminiQuotaError, GeminiAPIKeyError,
    MSG_QUOTA_EXHAUSTED, MSG_API_KEY_MISSING, is_quota_error_message,
)

logger = logging.getLogger(__name__)



# ── Intent categories ─────────────────────────────────────────────────────────

INTENT_CATEGORIES = {
    "work_experience":  ["worked", "job", "company", "employer", "role", "position", "internship", "employment"],
    "education":        ["studied", "degree", "university", "college", "school", "gpa", "academic", "course"],
    "projects":         ["project", "built", "developed", "created", "implemented", "made"],
    "skills":           ["skill", "know", "expertise", "proficient", "good at", "technology", "tech stack"],
    "technologies":     ["python", "java", "react", "aws", "machine learning", "deep learning", "nlp", "framework", "tool", "library"],
    "research":         ["research", "paper", "publication", "study", "experiment", "thesis", "published"],
    "achievements":     ["achievement", "award", "accomplishment", "won", "recognition", "honor", "impact", "result"],
    "biography":        ["tell me", "who is", "about", "background", "overview", "summary", "describe"],
    "opinions":         ["think", "believe", "opinion", "preference", "favorite", "like", "recommend"],
}

# For intent → retrieval strategy mapping
# "kg_first"     → extract graph entities, use KG local context primarily
# "vector_first" → narrative search, use vector chunks primarily
# "hybrid"       → combine both equally
INTENT_STRATEGY = {
    "work_experience":  "hybrid",
    "education":        "kg_first",
    "projects":         "hybrid",
    "skills":           "kg_first",
    "technologies":     "kg_first",
    "research":         "vector_first",
    "achievements":     "hybrid",
    "biography":        "vector_first",
    "opinions":         "vector_first",
    "general":          "hybrid",
}

# ── State ─────────────────────────────────────────────────────────────────────

class QAState(TypedDict):
    query:            str
    person_name:      str
    intent:           str
    strategy:         str
    retrieval_result: Optional[Dict[str, Any]]
    answer:           str
    key_facts:        List[str]
    citations:        List[str]
    confidence:       float
    should_refuse:    bool
    memo_facts:       Annotated[List[str], operator.add]
    chat_history:     Annotated[List[Dict], operator.add]


# ── Node 1: Intent classification ─────────────────────────────────────────────

def intent_node(state: QAState) -> dict:
    """Classify the query into a category and select retrieval strategy."""
    query_lower = state["query"].lower()
    scores: Dict[str, int] = {}

    for category, keywords in INTENT_CATEGORIES.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score:
            scores[category] = score

    intent = max(scores, key=scores.get) if scores else "biography"
    strategy = INTENT_STRATEGY.get(intent, "hybrid")

    return {"intent": intent, "strategy": strategy}


# ── Node 2: Retrieval ─────────────────────────────────────────────────────────

def retrieve_node(state: QAState) -> dict:
    """
    Run hybrid retrieval, weighting KG vs vector based on the chosen strategy.
    """
    result = hybrid_retrieve(
        state["query"],
        person_name=state["person_name"],
        strategy=state.get("strategy", "hybrid"),
    )
    return {"retrieval_result": result}


# ── Node 3: Answer generation ─────────────────────────────────────────────────

def answer_node(state: QAState) -> dict:
    """Generate a grounded, structured biographer response using Gemini."""
    retrieval = state["retrieval_result"]

    # ── Refusal path ──────────────────────────────────────────────────────────
    if retrieval["should_refuse"]:
        return {
            "answer": (
                f"I don't have enough information in {state['person_name']}'s profile to "
                f"answer that confidently (confidence: {retrieval['confidence']:.0%}). "
                "Try rephrasing or ask about a topic with more available data."
            ),
            "key_facts": [],
            "citations": [],
            "confidence": retrieval["confidence"],
            "should_refuse": True,
        }

    # ── Build context ─────────────────────────────────────────────────────────
    chunks_text      = format_chunks_for_prompt(retrieval["chunks"])
    graph_ctx        = retrieval.get("graph_context", "")
    community_ctx    = retrieval.get("community_context", "")
    memo_facts       = state.get("memo_facts", [])

    context_parts = []
    if graph_ctx:
        context_parts.append(f"KNOWLEDGE GRAPH CONTEXT:\n{graph_ctx}")
    if community_ctx:
        context_parts.append(f"COMMUNITY GRAPH CONTEXT:\n{community_ctx}")
    if chunks_text:
        context_parts.append(f"RETRIEVED DOCUMENT KNOWLEDGE:\n{chunks_text}")
    if memo_facts:
        context_parts.append(
            "CONFIRMED FACTS FROM THIS CONVERSATION:\n"
            + "\n".join(f"• {f}" for f in memo_facts[-10:])
        )

    context = "\n\n".join(context_parts)

    # ── Build citations ───────────────────────────────────────────────────────
    citations = []
    for chunk in retrieval["chunks"]:
        meta   = chunk.get("metadata", {})
        source = meta.get("source_label", meta.get("source", "document"))
        section = meta.get("section_index", "")
        label  = f"{source} §{section}" if section != "" else source
        if label not in citations:
            citations.append(label)
    if graph_ctx:
        citations.append("knowledge_graph")
    if community_ctx:
        citations.append("graph_community")

    # ── System prompt — biographer persona ────────────────────────────────────
    bio_text = _get_bio_text(state["person_name"])
    system_prompt = f"""You are an expert digital biographer for {state['person_name']}.
You have deep knowledge of their career, education, projects, skills, and achievements.
Your role is to answer questions about them in the style of a knowledgeable, articulate biographer.

Guidelines:
- Always ground answers in the provided context. Never hallucinate.
- Connect the dots across roles, projects, and skills to produce a smart narrative.
- Cite sources inline using [source] notation.
- If information is not in the context, say so clearly.
- Be concise but complete. Avoid padding.
- When listing technical items (tools, skills, technologies), be specific and accurate.

Biography Base:
{bio_text[:3000]}
"""

    # ── Chat history ──────────────────────────────────────────────────────────
    history = _format_history(state.get("chat_history", []))
    history_text = ""
    if history:
        history_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Biographer'}: {m['content']}"
            for m in history
        )

    # ── User prompt with structured output request ────────────────────────────
    user_prompt = f"""Context:
{context}

{f"Previous conversation:{chr(10)}{history_text}{chr(10)}" if history_text else ""}
Question: {state['query']}

Respond in the following JSON format (valid JSON only, no markdown):
{{
  "answer": "A full narrative answer in 2-4 sentences. Cite sources inline.",
  "key_facts": ["Specific fact 1", "Specific fact 2", "Specific fact 3"],
  "confidence_note": "High | Medium | Low — brief reason"
}}"""

    # ── Call LLM with retry + fallback ────────────────────────────────────────
    try:
        raw = gemini_generate(
            system_prompt + "\n\n" + user_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        answer    = parsed.get("answer", raw)
        key_facts = parsed.get("key_facts", [])
    except (GeminiQuotaError, GeminiAPIKeyError) as quota_exc:
        logger.warning("[answer_node] Quota/key error: %s", quota_exc)
        answer    = str(quota_exc)  # already user-friendly
        key_facts = []
    except json.JSONDecodeError:
        # Non-JSON response — use raw text
        answer    = raw if 'raw' in dir() else MSG_QUOTA_EXHAUSTED
        key_facts = []
    except Exception as exc:
        logger.error("[answer_node] Unexpected error: %s", exc)
        # Retry as plain text
        try:
            answer = gemini_generate(system_prompt + "\n\n" + user_prompt)
            key_facts = []
        except Exception as exc2:
            logger.error("[answer_node] Fallback also failed: %s", exc2)
            answer    = MSG_QUOTA_EXHAUSTED
            key_facts = []

    return {
        "answer":       answer,
        "key_facts":    key_facts,
        "citations":    citations,
        "confidence":   retrieval["confidence"],
        "should_refuse": False,
        "chat_history": [
            {"role": "user",      "content": state["query"]},
            {"role": "assistant", "content": answer},
        ],
    }


# ── Node 4: Memo persistence ──────────────────────────────────────────────────

def memo_node(state: QAState) -> dict:
    """Persist confirmed high-confidence facts across sessions via SqliteSaver."""
    if state.get("should_refuse") or not state.get("answer"):
        return {}
    if state.get("confidence", 0) < 0.5:
        return {}

    answer_snippet = state["answer"][:200].replace("\n", " ")
    memo_entry = f"Q: {state['query'][:80]} | A: {answer_snippet}"
    return {"memo_facts": [memo_entry]}


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_qa_graph():
    workflow = StateGraph(QAState)
    workflow.add_node("intent",   intent_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("answer",   answer_node)
    workflow.add_node("memo",     memo_node)

    workflow.set_entry_point("intent")
    workflow.add_edge("intent",   "retrieve")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer",   "memo")
    workflow.add_edge("memo",     END)

    import sqlite3
    conn   = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory)


_qa_graph = None


def get_qa_graph():
    global _qa_graph
    if _qa_graph is None:
        _qa_graph = build_qa_graph()
    return _qa_graph


# ── Public API ────────────────────────────────────────────────────────────────

async def ask_question(
    query: str,
    person_name: str,
    session_id: str = "default",
) -> Dict[str, Any]:
    """
    Ask a question about the person. Returns full structured result.
    Uses thread_id for cross-session memory via SqliteSaver.
    """
    graph  = get_qa_graph()
    config = {"configurable": {"thread_id": f"{person_name}_{session_id}"}}

    initial_state: QAState = {
        "query":            query,
        "person_name":      person_name,
        "intent":           "general",
        "strategy":         "hybrid",
        "retrieval_result": None,
        "answer":           "",
        "key_facts":        [],
        "citations":        [],
        "confidence":       0.0,
        "should_refuse":    False,
        "memo_facts":       [],
        "chat_history":     [],
    }

    result = await graph.ainvoke(initial_state, config=config)

    return {
        "answer":        result["answer"],
        "key_facts":     result.get("key_facts", []),
        "citations":     result.get("citations", []),
        "confidence":    result.get("confidence", 0.0),
        "intent":        result.get("intent", "general"),
        "strategy":      result.get("strategy", "hybrid"),
        "should_refuse": result.get("should_refuse", False),
        "sources_used":  result.get("retrieval_result", {}).get("sources_used", []),
        "query_entities": result.get("retrieval_result", {}).get("query_entities", []),
        "cache_stats":   get_cache_stats().to_dict(),
    }


async def stream_answer(
    query: str,
    person_name: str,
    session_id: str = "default",
) -> AsyncGenerator[str, None]:
    """
    Streaming version — yields SSE-compatible JSON events.
    Events: "intent" → "retrieval" → "token" → "done"
    """
    # Step 1: intent
    q_lower = query.lower()
    scores  = {cat: sum(1 for kw in kws if kw in q_lower) for cat, kws in INTENT_CATEGORIES.items()}
    intent  = max(scores, key=scores.get) if any(scores.values()) else "biography"
    strategy = INTENT_STRATEGY.get(intent, "hybrid")

    yield json.dumps({"type": "intent", "intent": intent, "strategy": strategy})
    yield "\n"

    # Step 2: retrieval
    retrieval = hybrid_retrieve(query, person_name=person_name, strategy=strategy)

    # Build clean source badges (Resume, LinkedIn, GitHub, Website, Interview only)
    SOURCE_LABEL_MAP = {
        "resume": "Resume", "linkedin": "LinkedIn", "github": "GitHub",
        "website": "Website", "interview": "Interview", "knowledge_graph": "Knowledge Graph",
    }
    raw_sources = set()
    for c in retrieval["chunks"]:
        sl = c.get("metadata", {}).get("source_label", "").lower()
        for key in SOURCE_LABEL_MAP:
            if key in sl:
                raw_sources.add(SOURCE_LABEL_MAP[key])
                break
        else:
            if sl:
                raw_sources.add(sl.title())
    if retrieval.get("graph_context"):
        raw_sources.add("Knowledge Graph")
    clean_citations = sorted(raw_sources)

    yield json.dumps({
        "type":          "retrieval",
        "confidence":    retrieval["confidence"],
        "sources_used":  retrieval["sources_used"],
        "query_entities": retrieval.get("query_entities", []),
        "citations":     clean_citations,
        "should_refuse": retrieval["should_refuse"],
    })
    yield "\n"

    if retrieval["should_refuse"]:
        yield json.dumps({
            "type": "token",
            "text": f"I don't have enough information to answer that confidently (confidence: {retrieval['confidence']:.0%}).",
        })
        yield "\n"
        yield json.dumps({"type": "done"})
        yield "\n"
        return

    # Step 3: build prompt & stream
    chunks_text   = format_chunks_for_prompt(retrieval["chunks"])
    graph_ctx     = retrieval.get("graph_context", "")
    community_ctx = retrieval.get("community_context", "")

    context_parts = []
    if graph_ctx:
        context_parts.append(f"KNOWLEDGE GRAPH CONTEXT:\n{graph_ctx}")
    if community_ctx:
        context_parts.append(f"COMMUNITY GRAPH CONTEXT:\n{community_ctx}")
    if chunks_text:
        context_parts.append(f"DOCUMENT KNOWLEDGE:\n{chunks_text}")
    context = "\n\n".join(context_parts)

    bio_text = _get_bio_text(person_name)
    system_text = f"""You are a digital biography agent for {person_name}.

Your job is to answer questions about {person_name} using only the provided context from their resume, LinkedIn, GitHub, websites, interviews, and knowledge graph.

Guidelines:
- Be conversational, warm, and concise. Write like an expert who personally knows this person.
- Prioritize correctness over creativity. Only state what is in the context.
- NEVER expose retrieval metadata, chunk IDs, graph queries, section numbers, or any internal system details.
- NEVER use citation notation like [source] or [resume §2] in your answer text.
- If information is unavailable, say so honestly and naturally.
- Summarize information in flowing prose — avoid bullet-point lists unless listing multiple distinct items.
- Make the interaction feel like speaking with an expert who knows this person's history, projects, skills, and achievements intimately.
- Keep answers focused and 2-5 sentences unless more detail is clearly needed.

Background context about {person_name}:
{bio_text[:2000]}"""

    user_msg = f"""Context about {person_name}:
{context}

Question: {query}

Answer naturally and conversationally. Do not mention sources, chunk IDs, or metadata in your answer."""
    full_prompt = system_text + "\n\n" + user_msg

    client_for_followups = None  # gemini_stream handles its own client internally
    full_answer = ""
    is_error = False

    try:
        for text_chunk in gemini_stream(full_prompt):
            if is_quota_error_message(text_chunk):
                # Surface as a special error event so UI can style it
                is_error = True
                full_answer = text_chunk
                yield json.dumps({"type": "token", "text": text_chunk, "is_quota_error": True})
                yield "\n"
            else:
                full_answer += text_chunk
                yield json.dumps({"type": "token", "text": text_chunk})
                yield "\n"
    except Exception as exc:
        logger.error("[stream_answer] Unexpected error during streaming: %s", exc)
        is_error = True
        full_answer = MSG_QUOTA_EXHAUSTED
        yield json.dumps({"type": "token", "text": MSG_QUOTA_EXHAUSTED, "is_quota_error": True})
        yield "\n"

    # Generate follow-up questions (skip if we hit an error)
    follow_up_questions = []
    if not is_error and full_answer:
        follow_up_questions = _generate_follow_ups(person_name, query, full_answer)

    yield json.dumps({
        "type":           "done",
        "intent":         intent,
        "strategy":       strategy,
        "follow_ups":     follow_up_questions,
        "is_quota_error": is_error,
        "cache_stats":    get_cache_stats().to_dict(),
    })
    yield "\n"


# ── Profile summary (for chat UI header) ─────────────────────────────────────

def get_profile_summary(person_name: str) -> Dict[str, Any]:
    """
    Build a structured profile card from the knowledge graph.
    Returns: name, current_role, companies, education, top_skills, notable_projects
    """
    graph = get_graph()
    nodes = dict(graph.g.nodes(data=True))

    def by_type(t):
        return [d["name"] for _, d in graph.g.nodes(data=True) if d.get("type") == t]

    companies   = by_type("COMPANY")
    roles       = by_type("ROLE")
    skills      = by_type("SKILL")[:8]
    tools       = by_type("TOOL")[:5]
    frameworks  = by_type("FRAMEWORK")[:5]
    projects    = by_type("PROJECT")[:6]
    degrees     = by_type("DEGREE")
    achievements = by_type("ACHIEVEMENT")[:4]
    technologies = by_type("TECHNOLOGY")[:5]

    # Current role: pick the first ROLE connected to a company (or just first role)
    current_role = roles[0] if roles else "Professional"
    current_company = companies[0] if companies else ""

    return {
        "name":            person_name,
        "current_role":    current_role,
        "current_company": current_company,
        "companies":       companies,
        "education":       degrees,
        "top_skills":      skills,
        "tools":           tools,
        "frameworks":      frameworks,
        "technologies":    technologies,
        "notable_projects": projects,
        "achievements":    achievements,
        "graph_stats": {
            "total_nodes": graph.g.number_of_nodes(),
            "total_edges": graph.g.number_of_edges(),
        },
    }


def get_suggested_questions(person_name: str) -> List[str]:
    """Generate contextual suggested questions from the knowledge graph."""
    graph  = get_graph()
    companies = [d["name"] for _, d in graph.g.nodes(data=True) if d.get("type") == "COMPANY"]
    projects  = [d["name"] for _, d in graph.g.nodes(data=True) if d.get("type") == "PROJECT"]
    skills    = [d["name"] for _, d in graph.g.nodes(data=True) if d.get("type") == "SKILL"]

    suggestions = [
        f"Tell me about {person_name}'s background.",
        f"What companies has {person_name} worked for?",
        f"What are {person_name}'s strongest technical skills?",
        f"What projects have they worked on?",
        f"What technologies does {person_name} use most?",
    ]

    if companies:
        suggestions.append(f"What did {person_name} do at {companies[0]}?")
    if projects:
        suggestions.append(f"Tell me about the {projects[0]} project.")
    if skills:
        suggestions.append(f"How did {person_name} develop expertise in {skills[0]}?")

    return suggestions[:8]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_bio_text(person_name: str) -> str:
    """Get combined ingested text as stable bio prefix for the LLM."""
    try:
        from backend.ingestion.document import get_all_chunks
        chunks  = get_all_chunks()
        combined = "\n\n".join(c["text"] for c in chunks[:30])
        return combined[:6000]
    except Exception:
        return f"Professional profile for {person_name}."


def _format_history(history: List[Dict]) -> List[Dict]:
    """Return last 6 turns of chat history."""
    return [
        {"role": h["role"], "content": h["content"]}
        for h in history[-6:]
        if h.get("role") in ("user", "assistant")
    ]


def _generate_follow_ups(person_name: str, question: str, answer: str) -> List[str]:
    """
    Generate 3 contextual follow-up questions based on what was just discussed.
    Uses the resilient gemini_generate — falls back to static questions on any error.
    """
    fallback = [
        "Tell me more about this project",
        "What technologies were used?",
        "What was their biggest achievement here?",
    ]
    try:
        prompt = f"""You are helping someone explore a career biography.
The user just asked: "{question}"
The biography agent answered: "{answer[:500]}"

Generate exactly 3 short, natural follow-up questions someone might ask next to learn more about {person_name}.
Rules:
- Each question must be on its own line, no numbering or bullets.
- Questions must be short (under 12 words).
- Make them specific and relevant to what was just discussed.
- Do NOT repeat the original question.
- Do NOT ask about things not mentioned in the answer.

Output only the 3 questions, one per line."""

        text = gemini_generate(prompt)
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        questions = [l.lstrip("0123456789.-) ") for l in lines[:3]]
        return questions if len(questions) == 3 else fallback
    except Exception as exc:
        logger.warning("[_generate_follow_ups] Failed: %s", exc)
        return fallback
