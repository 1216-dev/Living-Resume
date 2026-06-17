import json
import logging
import re
from typing import Dict, Any

from google.genai import types

from backend.config import GEMINI_MODEL, GEMINI_API_KEY
from backend.agents.gemini_client import gemini_generate

logger = logging.getLogger(__name__)


def process_crawled_content(url: str, markdown: str) -> Dict[str, Any]:
    """
    Takes raw, potentially noisy markdown scraped from a URL and uses
    an LLM to extract a clean summary, timeline, and keywords.
    """
    if not markdown.strip():
        return {"summary": "", "timeline": [], "keywords": []}

    prompt = f"""You are an expert Data Extractor analyzing scraped website content from: {url}.
The text below is raw markdown. Please extract the most important professional and biographical information into strict JSON.

RAW MARKDOWN:
{markdown[:15000]}

Please extract all available information and structure it strictly into the following JSON format.
{{
  "summary": "A clean, well-written, 2-3 paragraph professional summary of the content on this page. Remove all navigation/footer noise.",
  "timeline": [
    {{
      "start_date": "YYYY-MM",
      "end_date": "YYYY-MM (or Present)",
      "title": "Role / Degree / Project Name",
      "organization": "Company / University",
      "description": "Detailed bullet points or summary of what was achieved or done."
    }}
  ],
  "keywords": [
    "keyword1", "keyword2"
  ]
}}

Rules:
1. "summary" must be pure text, without markdown formatting.
2. If there is no timeline data, return an empty array for "timeline".
3. Extract core competencies, skills, and important entities as "keywords".
4. Output ONLY the raw JSON object.
"""

    try:
        raw = gemini_generate(
            prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        return data
    except Exception as e:
        logger.error("[CrawlerAgent] Failed to extract insights from %s: %s", url, e)
        return {"summary": markdown[:2000], "timeline": [], "keywords": []}

