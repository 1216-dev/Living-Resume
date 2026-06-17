import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any

from google import genai
from google.genai import types

from backend.config import GEMINI_MODEL, GEMINI_API_KEY
from backend.agents.gemini_client import gemini_generate, _get_client

logger = logging.getLogger(__name__)


def extract_document_insights(file_path: str, person_name: str) -> Dict[str, Any]:
    """
    Uses Gemini's multimodal capabilities (or text extraction fallback)
    to perform deep extraction of a document (Resume, CV, etc).
    Extracts a strict chronological timeline and a master list of keywords/skills.
    """
    # Need raw client for file upload API
    client = _get_client()

    path = Path(file_path)
    mime_type = "text/plain"
    if path.suffix.lower() == ".pdf":
        mime_type = "application/pdf"
    # Upload file to Gemini for multimodal analysis
    uploaded_file = client.files.upload(file=str(path))

    prompt = f"""You are an expert HR and Technical Recruiter analyzing a document about {person_name}.
Please extract all available information from the document and structure it strictly into the following JSON format.

{{
  "candidate_name": "First Last",
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
1. Extract the candidate's actual name. If possible, provide their full name.
2. Extract EVERY SINGLE skill, technology, tool, methodology, and domain as a keyword. Be extremely thorough.
3. Build a complete chronological timeline of all roles, education, and projects mentioned.
4. If dates are fuzzy, do your best to approximate or leave as the raw string from the document.
5. Output ONLY the raw JSON object, no markdown blocks.
"""

    try:
        # For multimodal (file + text), we call the raw client directly
        # but still benefit from the resilient client's model config
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        return data
    except Exception as e:
        logger.error("[DocumentAgent] Failed to extract insights: %s", e)
        return {"timeline": [], "keywords": []}
    finally:
        # Clean up the file from Gemini storage
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

