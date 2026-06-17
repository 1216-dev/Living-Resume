"""
agents/interview_agent.py  —  Persona-Aware Biography Interview Pipeline
─────────────────────────────────────────────────────────────────────────

The interviewer persona is a blend of:
  • Podcast host       — conversational, warm, curious, lets silence breathe
  • Senior recruiter   — uncovers accomplishments, scope, impact, metrics
  • Career coach       — surfaces growth, lessons, motivations, values
  • Research journalist — digs for specifics: names, dates, decisions, outcomes

Pipeline per turn:
  User answer
    ↓
  [analyse_answer]   — extract entities + identify gaps + score completeness
    ↓
  [generate_question] — craft single best follow-up OR pivot to new topic
    ↓
  [ingest_facts]     — persist Q&A + entities into vector store + graph
    ↓
  Return next question + extracted entities + progress

Key design principles:
  • Full conversation history always in context → no repetition
  • Follow-up priority: if answer mentions something interesting, dig deeper
  • Depth over breadth: complete one thread before moving on
  • Entity-aware: recognised triggers immediately prompt exploration
"""

import json
import logging
import os
import re
import sqlite3
from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator

from google.genai import types
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from backend.config import SQLITE_PATH, TOPIC_AREAS, MAX_INTERVIEW_TURNS
from backend.ingestion.document import ingest_text
from backend.knowledge.graph import get_graph
from backend.knowledge.bm25_index import rebuild_bm25_from_chroma
from backend.agents.gemini_client import gemini_generate

logger = logging.getLogger(__name__)


# ── Persona system prompt ─────────────────────────────────────────────────────

INTERVIEWER_PERSONA = """You are an AI interviewer whose sole goal is to build a rich, vivid digital biography of the person you are talking to.

Your persona is a blend of:
- An experienced podcast host (Lex Fridman, Tim Ferriss) — warm, patient, genuinely curious, lets answers breathe
- A senior technical recruiter — uncovers real scope, metrics, ownership, and impact
- A career coach — surfaces growth mindset, lessons learned, motivations, values
- A research journalist — hunts for specifics: company names, dates, decisions, outcomes, turning points

Your core rules:
1. Ask EXACTLY ONE question per turn. Never combine two questions.
2. Build naturally on what the person just said. If they mentioned a project, technology, challenge, or achievement — explore it before moving on.
3. Prioritise DEPTH over BREADTH. Go deep on one thread before switching topics.
4. Be warm and conversational. No bureaucratic or generic phrasing.
5. NEVER ask something already answered. You have the full conversation history.
6. When someone mentions a TRIGGER ENTITY — explore it immediately:
   - Company / startup / project → scope, team size, your role, outcome
   - Technology / framework → why chosen, what built with it, lessons
   - Achievement / metric → how measured, what you did specifically, impact
   - Failure / challenge → what happened, what you tried, what you learned
   - Person (mentor, colleague) → what you learned from them
   - Decision / turning point → what drove the decision, outcome, regret or pride?
7. Use the conversation so far to avoid repetition and to reference earlier answers.
8. Keep questions under 25 words. Conversational, not essay-prompts.
9. Occasionally acknowledge what they said before asking (e.g. "That's a bold move — what made you choose…").
10. End the interview naturally when all major topics are richly covered."""

# ── Topics and their trigger signals ─────────────────────────────────────────

TOPIC_AREAS_EXT = [
    "career_journey",
    "work_experience",
    "education",
    "projects",
    "technical_skills",
    "leadership",
    "research_and_publications",
    "achievements",
    "failures_and_lessons",
    "personal_interests",
    "future_goals",
]

TOPIC_OPENERS = {
    "career_journey":            "How did you first get into this field?",
    "work_experience":           "Walk me through your most impactful role so far.",
    "education":                 "Tell me about your educational background — what shaped your thinking the most?",
    "projects":                  "What's a project you're genuinely proud of, and why?",
    "technical_skills":          "What's a technology you've gone really deep on, and what does that depth look like?",
    "leadership":                "Tell me about a time you led a team or initiative — what was the hardest part?",
    "research_and_publications": "Have you done any research, written papers, or contributed to open source?",
    "achievements":              "What's an achievement that you don't think shows up well on a resume but you're proud of?",
    "failures_and_lessons":      "Tell me about a time something didn't go as planned — what did you learn?",
    "personal_interests":        "What do you work on outside of your job that excites you?",
    "future_goals":              "Where do you see yourself heading in the next few years?",
}

TOPIC_LABELS = {
    "career_journey":            "🚀 Career Journey",
    "work_experience":           "💼 Work Experience",
    "education":                 "🎓 Education",
    "projects":                  "🔧 Projects",
    "technical_skills":          "⚙️ Technical Skills",
    "leadership":                "👥 Leadership",
    "research_and_publications": "📄 Research",
    "achievements":              "🏆 Achievements",
    "failures_and_lessons":      "💡 Lessons Learned",
    "personal_interests":        "🎯 Personal Interests",
    "future_goals":              "🌟 Future Goals",
}


# ── State ─────────────────────────────────────────────────────────────────────

class InterviewState(TypedDict):
    person_name: str
    covered_topics: Annotated[List[str], operator.add]
    turn_count: int
    conversation_history: Annotated[List[Dict], operator.add]
    current_question: str
    last_answer: str
    extracted_entities: Annotated[List[Dict], operator.add]
    pending_followup: str            # If set, ask this before switching topics
    current_topic: str
    is_complete: bool
    consecutive_followups: int       # Limit follow-up depth to avoid rabbit holes


# ── Answer analysis ───────────────────────────────────────────────────────────

def analyse_answer(person_name: str, question: str, answer: str, history: List[Dict]) -> Dict[str, Any]:
    """
    Deep analysis of the answer:
    - Extract entities + relationships
    - Detect trigger entities that deserve follow-up
    - Identify what topics are implicitly covered
    - Suggest the single best next question
    """
    history_txt = "\n".join(
        f"{'Interviewer' if h['role'] == 'assistant' else person_name}: {h['content']}"
        for h in history[-12:]
    )

    prompt = f"""{INTERVIEWER_PERSONA}

You just conducted this interview exchange:

--- CONVERSATION HISTORY ---
{history_txt}
--- END HISTORY ---

Latest exchange:
Interviewer: {question}
{person_name}: {answer}

Now perform a deep analysis. Return ONLY valid JSON matching exactly this schema:

{{
  "entities": [
    {{
      "type": "COMPANY|ROLE|PROJECT|SKILL|TECHNOLOGY|FRAMEWORK|TOOL|DEGREE|ACHIEVEMENT|CHALLENGE|PUBLICATION|PERSON|LOCATION|DATASET|MOTIVATION|VALUE|INTEREST",
      "name": "exact name as mentioned",
      "context": "1-sentence context from the answer",
      "is_trigger": true/false
    }}
  ],
  "relationships": [
    {{
      "from": "{person_name}",
      "relation": "WORKED_AT|FOUNDED|BUILT|ACHIEVED|STUDIED_AT|USED|LED|MENTORED_BY|COLLABORATED_WITH|PUBLISHED|INTERESTED_IN|ASPIRES_TO",
      "to": "entity name"
    }}
  ],
  "topics_covered": ["career_journey|work_experience|education|projects|technical_skills|leadership|research_and_publications|achievements|failures_and_lessons|personal_interests|future_goals"],
  "trigger_entities": ["list entity names that are most interesting to explore further"],
  "missing_depth": "1 sentence describing what specific detail is still unknown about the most interesting part of this answer",
  "suggested_followup": "The single best follow-up question to ask RIGHT NOW (max 20 words). If nothing important to follow up on, return empty string.",
  "followup_reasoning": "Why this follow-up question is the highest value next question"
}}

Be extremely thorough in entity extraction. If the answer mentions a startup, a metric like '40% improvement', a specific technology, a team size, a decision — capture it all."""

    try:
        raw = gemini_generate(
            prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as exc:
        logger.warning("[InterviewAgent] analyse_answer failed: %s", exc)
        return {
            "entities": [],
            "relationships": [],
            "topics_covered": [],
            "trigger_entities": [],
            "missing_depth": "",
            "suggested_followup": "",
            "followup_reasoning": "",
        }


# ── Question generation ───────────────────────────────────────────────────────

def generate_next_question(
    person_name: str,
    history: List[Dict],
    next_topic: str,
    pending_followup: str = "",
    covered_topics: List[str] = None,
) -> str:
    """
    Generate the single best next question using full conversation context.
    If there's a pending follow-up from analyse_answer, prefer it.
    Otherwise transition to the next uncovered topic naturally.
    """
    if pending_followup:
        return pending_followup

    covered_topics = covered_topics or []
    history_txt = "\n".join(
        f"{'Interviewer' if h['role'] == 'assistant' else person_name}: {h['content']}"
        for h in history[-14:]
    )

    opener = TOPIC_OPENERS.get(next_topic, f"Tell me about your {next_topic.replace('_', ' ')}.")
    covered_str = ", ".join(covered_topics) if covered_topics else "none yet"

    prompt = f"""{INTERVIEWER_PERSONA}

You are interviewing {person_name}. Here is the conversation so far:

{history_txt}

Topics already well covered: {covered_str}
Next topic to explore: {next_topic}
Suggested opener for this topic: "{opener}"

Generate EXACTLY ONE interview question to ask next.

Rules:
- Transition NATURALLY from what was just discussed to the new topic if possible
- If the conversation is just starting on this topic, adapt the suggested opener to feel natural given the conversation flow
- Do NOT repeat anything already asked or answered
- Keep it under 25 words
- Sound like a brilliant podcast host, not a form

Return ONLY the question text. No preamble. No explanation."""

    try:
        question = gemini_generate(prompt)
        return question.strip().strip('"')
    except Exception as exc:
        logger.warning("[InterviewAgent] generate_next_question failed: %s", exc)
        return opener


# ── Entity ingestion ──────────────────────────────────────────────────────────

def ingest_exchange(person_name: str, question: str, answer: str, topic: str, entities_data: Dict):
    """Persist Q&A and extracted entities into vector store + knowledge graph."""
    source_label = f"interview_{topic}"
    try:
        ingest_text(
            text=f"Q: {question}\nA: {answer}",
            source_label=source_label,
            person_name=person_name,
            metadata_extra={"interview_topic": topic},
        )
    except Exception as exc:
        logger.warning("[InterviewAgent] ingest_text failed: %s", exc)

    try:
        graph = get_graph()
        graph.add_entities_from_extraction(entities_data, source=source_label)
        graph.compute_communities()
        graph.save()
        rebuild_bm25_from_chroma()
    except Exception as exc:
        logger.warning("[InterviewAgent] graph update failed: %s", exc)


# ── Pre-coverage detection ────────────────────────────────────────────────────

def detect_pre_covered_topics(person_name: str) -> List[str]:
    """Check which topics are already known from ingested documents."""
    try:
        from backend.ingestion.document import get_all_chunks
        chunks = get_all_chunks()
        if not chunks:
            return []
        all_text = " ".join(c["text"] for c in chunks).lower()
        signals = {
            "career_journey":    ["career", "journey", "started", "path", "field"],
            "work_experience":   ["worked at", "role", "position", "engineer", "manager", "intern", "company"],
            "education":         ["university", "college", "degree", "bachelor", "master", "phd", "studied"],
            "projects":          ["built", "developed", "created", "project", "system", "application"],
            "technical_skills":  ["python", "aws", "react", "sql", "machine learning", "tensorflow", "pytorch"],
            "leadership":        ["led", "managed", "mentored", "team lead", "leadership"],
            "achievements":      ["achieved", "award", "recognition", "milestone", "improved", "reduced", "increased"],
            "failures_and_lessons": ["challenge", "problem", "difficulty", "overcame", "failed", "mistake"],
            "personal_interests":   ["hobby", "interest", "passion", "outside work", "side project"],
            "future_goals":         ["goal", "aspire", "plan", "next step", "future", "vision"],
        }
        return [
            topic for topic, kws in signals.items()
            if sum(1 for kw in kws if kw in all_text) >= 3
        ]
    except Exception:
        return []


# ── Public API ────────────────────────────────────────────────────────────────

# In-memory session store (keyed by session_id)
# Holds: conversation_history, covered_topics, turn_count, current_topic, consecutive_followups
_sessions: Dict[str, Dict] = {}


async def start_or_continue_interview(
    person_name: str,
    session_id: str,
    human_answer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stateful interview driver.

    First call (human_answer=None):
      - Detect pre-covered topics from docs
      - Generate opening question
      - Return question + progress

    Subsequent calls (human_answer=<text>):
      - Analyse the answer (extract entities, detect follow-ups)
      - Decide: follow up on this answer, or pivot to next topic
      - Ingest Q&A + entities into knowledge base
      - Return next question + extracted entities + progress
    """
    key = f"{person_name}_{session_id}"

    # ── First call ─────────────────────────────────────────────────────────────
    if human_answer is None:
        pre_covered = detect_pre_covered_topics(person_name)
        remaining = [t for t in TOPIC_AREAS_EXT if t not in pre_covered]
        first_topic = remaining[0] if remaining else TOPIC_AREAS_EXT[0]

        _sessions[key] = {
            "person_name":           person_name,
            "covered_topics":        list(pre_covered),
            "turn_count":            0,
            "conversation_history":  [],
            "current_topic":         first_topic,
            "pending_followup":      "",
            "consecutive_followups": 0,
        }

        # Generate opening question
        opening = TOPIC_OPENERS.get(first_topic, f"Tell me about yourself, {person_name}.")
        # If no docs, use a warm personalised opener
        if not pre_covered:
            opening = f"It's great to meet you, {person_name}! Let's start at the beginning — how did you first get into your field?"
        else:
            # Docs exist — acknowledge and bridge to a gap
            opening = generate_next_question(
                person_name, [], first_topic, covered_topics=pre_covered
            )

        _sessions[key]["conversation_history"].append(
            {"role": "assistant", "content": opening}
        )
        _sessions[key]["turn_count"] = 1

        return {
            "question":              opening,
            "is_complete":           False,
            "covered_topics":        list(pre_covered),
            "remaining_topics":      remaining[1:],
            "progress_pct":          int(len(pre_covered) / len(TOPIC_AREAS_EXT) * 100),
            "pre_covered_from_docs": list(pre_covered),
            "topic_labels":          TOPIC_LABELS,
            "extracted_entities":    [],
        }

    # ── Subsequent calls ───────────────────────────────────────────────────────
    session = _sessions.get(key)
    if not session:
        # Session lost (restart) — re-init and recurse
        return await start_or_continue_interview(person_name, session_id, human_answer=None)

    last_question = session["conversation_history"][-1]["content"] if session["conversation_history"] else ""
    session["conversation_history"].append({"role": "user", "content": human_answer})

    # ── Analyse the answer
    analysis = analyse_answer(
        person_name,
        last_question,
        human_answer,
        session["conversation_history"],
    )

    entities        = analysis.get("entities", [])
    topics_covered  = analysis.get("topics_covered", [])
    followup        = analysis.get("suggested_followup", "").strip()
    trigger_entities = analysis.get("trigger_entities", [])

    # Update covered topics
    new_covered = list(set(session["covered_topics"] + topics_covered))
    session["covered_topics"] = new_covered

    # ── Ingest exchange
    ingest_exchange(
        person_name,
        last_question,
        human_answer,
        session["current_topic"],
        {"entities": entities, "relationships": analysis.get("relationships", [])},
    )

    # ── Decide next question
    remaining = [t for t in TOPIC_AREAS_EXT if t not in new_covered]
    turn = session["turn_count"] + 1
    session["turn_count"] = turn

    # Check completion
    if not remaining or turn >= MAX_INTERVIEW_TURNS:
        session["is_complete"] = True
        return {
            "question":           None,
            "is_complete":        True,
            "covered_topics":     new_covered,
            "remaining_topics":   [],
            "progress_pct":       100,
            "extracted_entities": entities,
            "topic_labels":       TOPIC_LABELS,
            "message": (
                f"What a conversation! I've now built a detailed picture of {person_name} across "
                f"{len(new_covered)} topic areas. The knowledge base is ready — switch to Chat to explore it."
            ),
        }

    # Follow-up logic: dig deeper if there's a good follow-up AND we haven't gone too deep
    consecutive = session.get("consecutive_followups", 0)
    use_followup = followup and consecutive < 3 and trigger_entities

    if use_followup:
        next_question = followup
        session["consecutive_followups"] = consecutive + 1
        # Keep current topic (still exploring the same thread)
    else:
        # Pivot to next uncovered topic
        next_topic = remaining[0]
        session["current_topic"] = next_topic
        session["consecutive_followups"] = 0
        next_question = generate_next_question(
            person_name,
            session["conversation_history"],
            next_topic,
            covered_topics=new_covered,
        )

    session["conversation_history"].append({"role": "assistant", "content": next_question})

    return {
        "question":           next_question,
        "is_complete":        False,
        "covered_topics":     new_covered,
        "remaining_topics":   remaining[1:] if not use_followup else remaining,
        "progress_pct":       int(len(new_covered) / len(TOPIC_AREAS_EXT) * 100),
        "extracted_entities": entities,
        "trigger_entities":   trigger_entities,
        "is_followup":        use_followup,
        "current_topic":      session["current_topic"],
        "topic_labels":       TOPIC_LABELS,
        "missing_depth":      analysis.get("missing_depth", ""),
        "facts_extracted":    len(entities),
    }


def get_topic_labels() -> Dict[str, str]:
    return TOPIC_LABELS


def get_topic_areas() -> List[str]:
    return TOPIC_AREAS_EXT
