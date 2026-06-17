"""
ingestion/linkedin_mcp.py
──────────────────────────
Dual-mode LinkedIn connector:

  Mode 1 — FastMCP server (live data via Proxycurl API)
    Start with: python -m backend.ingestion.linkedin_mcp
    Exposes MCP tools: get_linkedin_profile, get_linkedin_posts, get_github_repos

  Mode 2 — PDF export fallback
    Parses a downloaded LinkedIn PDF export.

GitHub data is fetched via public GitHub API (no auth needed for public profiles).
"""
import json
import os
from typing import Dict, Any, List, Optional

import httpx

from backend.config import PROXYCURL_API_KEY


# ── FastMCP Server ────────────────────────────────────────────────────────────

def create_linkedin_mcp_server():
    """
    Create a FastMCP server exposing LinkedIn + GitHub tools.
    Run standalone: uvicorn backend.ingestion.linkedin_mcp:mcp_app
    """
    try:
        from fastmcp import FastMCP
        mcp = FastMCP("Living Resume — LinkedIn Connector")

        @mcp.tool()
        async def get_linkedin_profile(linkedin_url: str) -> dict:
            """
            Fetch a LinkedIn profile via Proxycurl API.
            Returns structured profile data including experience, education, skills.
            """
            return await _proxycurl_fetch_profile(linkedin_url)

        @mcp.tool()
        async def get_linkedin_posts(linkedin_url: str, max_posts: int = 10) -> dict:
            """
            Fetch recent LinkedIn posts for a profile.
            Returns list of post texts for ingestion.
            """
            return await _proxycurl_fetch_posts(linkedin_url, max_posts)

        @mcp.tool()
        async def get_github_repos(username: str) -> dict:
            """
            Fetch GitHub repositories and contribution summary for a user.
            """
            return await fetch_github_data(username)

        return mcp
    except ImportError:
        return None


# ── Proxycurl API helpers ─────────────────────────────────────────────────────

async def _proxycurl_fetch_profile(linkedin_url: str) -> Dict[str, Any]:
    """Fetch LinkedIn profile via Proxycurl."""
    if not PROXYCURL_API_KEY:
        return {
            "available": False,
            "message": "PROXYCURL_API_KEY not set. Upload LinkedIn PDF export instead.",
            "profile_url": linkedin_url,
        }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://nubela.co/proxycurl/api/v2/linkedin",
                headers={"Authorization": f"Bearer {PROXYCURL_API_KEY}"},
                params={"url": linkedin_url, "use_cache": "if-present"},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "available": True,
                "source": "proxycurl",
                "profile": data,
                "text": _profile_to_text(data),
            }
    except Exception as e:
        return {"available": False, "error": str(e), "profile_url": linkedin_url}


async def _proxycurl_fetch_posts(linkedin_url: str, max_posts: int = 10) -> Dict[str, Any]:
    """Fetch LinkedIn posts via Proxycurl."""
    if not PROXYCURL_API_KEY:
        return {"available": False, "posts": []}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://nubela.co/proxycurl/api/v1/linkedin/post/search/person",
                headers={"Authorization": f"Bearer {PROXYCURL_API_KEY}"},
                params={"linkedin_profile_url": linkedin_url, "count": max_posts},
            )
            resp.raise_for_status()
            data = resp.json()
            posts = data.get("posts", [])
            return {
                "available": True,
                "posts": posts,
                "text": "\n\n".join(
                    f"[LinkedIn Post]\n{p.get('text', '')}" for p in posts[:max_posts]
                ),
            }
    except Exception as e:
        return {"available": False, "error": str(e), "posts": []}


def _profile_to_text(profile: dict) -> str:
    """Convert Proxycurl profile JSON → clean text for ingestion."""
    lines = []
    name = f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
    if name:
        lines.append(f"Name: {name}")
    if profile.get("headline"):
        lines.append(f"Headline: {profile['headline']}")
    if profile.get("summary"):
        lines.append(f"Summary: {profile['summary']}")
    if profile.get("city") or profile.get("country_full_name"):
        lines.append(f"Location: {profile.get('city', '')}, {profile.get('country_full_name', '')}")

    # Experience
    for exp in profile.get("experiences", []):
        company = exp.get("company", "")
        title = exp.get("title", "")
        desc = exp.get("description", "")
        start = exp.get("starts_at", {}) or {}
        end = exp.get("ends_at", {}) or {}
        period = f"{start.get('year', '?')} – {end.get('year', 'Present')}"
        lines.append(f"\nWork Experience: {title} at {company} ({period})")
        if desc:
            lines.append(f"  {desc[:300]}")

    # Education
    for edu in profile.get("education", []):
        school = edu.get("school", "")
        degree = edu.get("degree_name", "")
        field = edu.get("field_of_study", "")
        lines.append(f"\nEducation: {degree} in {field} at {school}")

    # Skills
    skills = [s.get("name", "") for s in profile.get("accomplishment_courses", [])]
    if profile.get("skills"):
        skills = profile["skills"][:30]
    if skills:
        lines.append(f"\nSkills: {', '.join(str(s) for s in skills)}")

    return "\n".join(lines)


# ── Public alias ──────────────────────────────────────────────────────────────
# Exposed for import in main.py
async def fetch_linkedin_profile(linkedin_url: str) -> Dict[str, Any]:
    """
    Public entry point for LinkedIn profile fetch.
    Tries Proxycurl API, falls back with a helpful message.
    """
    return await _proxycurl_fetch_profile(linkedin_url)


# ── GitHub API ────────────────────────────────────────────────────────────────

async def fetch_github_data(github_username: str) -> Dict[str, Any]:
    """
    Fetch GitHub repos + contribution data via GitHub public API.
    No auth required for public profiles.
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # User profile
            user_resp = await client.get(
                f"https://api.github.com/users/{github_username}",
                headers={"User-Agent": "living-resume-bot", "Accept": "application/vnd.github.v3+json"},
            )
            user_data = user_resp.json() if user_resp.status_code == 200 else {}

            # Repos
            repos_resp = await client.get(
                f"https://api.github.com/users/{github_username}/repos?sort=updated&per_page=20",
                headers={"User-Agent": "living-resume-bot", "Accept": "application/vnd.github.v3+json"},
            )
            repos = repos_resp.json() if repos_resp.status_code == 200 else []

        repo_summaries = []
        for r in repos[:15]:
            if isinstance(r, dict):
                repo_summaries.append({
                    "name": r.get("name", ""),
                    "description": r.get("description", ""),
                    "language": r.get("language", ""),
                    "stars": r.get("stargazers_count", 0),
                    "topics": r.get("topics", []),
                    "url": r.get("html_url", ""),
                    "updated": r.get("updated_at", ""),
                })

        # Format as ingestion text
        text_lines = [
            f"GitHub profile: {github_username}",
            f"Bio: {user_data.get('bio', '')}",
            f"Public repos: {user_data.get('public_repos', 0)}",
            f"Followers: {user_data.get('followers', 0)}",
            "",
        ]
        for repo in repo_summaries:
            text_lines.append(f"Repository: {repo['name']}")
            if repo["description"]:
                text_lines.append(f"  Description: {repo['description']}")
            if repo["language"]:
                text_lines.append(f"  Primary language: {repo['language']}")
            if repo["topics"]:
                text_lines.append(f"  Topics: {', '.join(repo['topics'])}")
            text_lines.append("")

        return {
            "source": "github_api",
            "available": True,
            "username": github_username,
            "repos": repo_summaries,
            "text": "\n".join(text_lines),
        }

    except Exception as e:
        return {"source": "github_api", "available": False, "error": str(e), "text": ""}


# ── LinkedIn PDF fallback ──────────────────────────────────────────────────────

def linkedin_pdf_to_text(file_path: str) -> str:
    """
    Parse a LinkedIn PDF export into clean text.
    LinkedIn exports have a fairly consistent structure.
    """
    try:
        import pdfplumber
        lines = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    lines.append(text)
        return "\n".join(lines)
    except ImportError:
        try:
            from llama_index.core import SimpleDirectoryReader
            docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
            return "\n".join(d.text for d in docs)
        except Exception as e:
            return f"[Could not parse LinkedIn PDF: {e}]"


# ── Standalone MCP server entrypoint ─────────────────────────────────────────

if __name__ == "__main__":
    mcp = create_linkedin_mcp_server()
    if mcp:
        mcp.run()
    else:
        print("FastMCP not installed. Run: pip install fastmcp")
