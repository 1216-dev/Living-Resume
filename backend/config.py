"""
Central config — reads .env and exposes typed settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
PROXYCURL_API_KEY: str = os.getenv("PROXYCURL_API_KEY", "")
TINYFISH_API_KEY: str = os.getenv("TINYFISH_API_KEY", "")

# ── Storage paths ─────────────────────────────────────────────────────────────
# config.py lives at backend/config.py → parent is backend/ → parent.parent is living-resume/
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_PATH = str(DATA_DIR / "chroma")
GRAPH_DB_PATH = str(DATA_DIR / "graph.json")
SQLITE_PATH = str(DATA_DIR / "memory.db")
UPLOAD_DIR = str(DATA_DIR / "uploads")

# ── Optional website to crawl ─────────────────────────────────────────────────
CRAWL_TARGET_URL: str = os.getenv("CRAWL_TARGET_URL", "")

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_VECTOR = 5          # how many vector chunks to pull
TOP_K_BM25 = 5            # how many BM25 chunks to pull
TOP_K_GRAPH = 3           # how many graph-neighbour summaries to pull
FINAL_TOP_K = 6           # after RRF fusion, how many to send to LLM
CONFIDENCE_THRESHOLD = 0.35  # below this → "I don't have enough data"

# ── KV Cache ──────────────────────────────────────────────────────────────────
# cache_control is built-in for claude-sonnet-4-x — no beta flag needed.
# The stable bio prefix is# Caching 
# Note: Gemini Context Caching is omitted for now because it requires a 32,768 token minimum,
# which the bio prefix likely won't hit.
KV_CACHE_MIN_TOKENS = 32768   # Anthropic minimum for cache_control to activate

# ── Graph entity types ────────────────────────────────────────────────────────
ENTITY_TYPES = ["PERSON", "COMPANY", "ROLE", "SKILL", "PROJECT", "DEGREE", "LOCATION", "TOOL", "TECHNOLOGY", "ACHIEVEMENT", "PUBLICATION", "FRAMEWORK", "DATASET", "RESPONSIBILITY"]

# ── GraphRAG ──────────────────────────────────────────────────────────────────
GRAPH_COMMUNITY_MIN_SIZE = 2  # minimum nodes to form a named community

# ── Interview agent ───────────────────────────────────────────────────────────
MAX_INTERVIEW_TURNS = 20
TOPIC_AREAS = [
    "work_experience",
    "education",
    "skills_and_tools",
    "projects",
    "leadership",
    "challenges",
    "opinions_and_values",
]

# Ensure dirs exist
for _d in [CHROMA_DB_PATH, UPLOAD_DIR, str(DATA_DIR)]:
    Path(_d).mkdir(parents=True, exist_ok=True)
