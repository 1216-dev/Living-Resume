"""
ingestion/web_crawler.py
─────────────────────────
Crawls a portfolio / personal website using Crawl4AI.
Returns clean markdown per page, then ingests into the same pipeline.
Handles JS-rendered SPAs, extracts clean markdown, follows internal links.
"""
import asyncio
from typing import Dict, Any, List
from urllib.parse import urljoin, urlparse


async def crawl_url(url: str) -> Dict[str, Any]:
    """
    Crawl a single URL with Crawl4AI. Returns {url, markdown, title, success}.
    Falls back gracefully if crawl4ai isn't available or JS rendering fails.
    """
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        browser_cfg = BrowserConfig(headless=True, verbose=False)
        run_cfg = CrawlerRunConfig(
            word_count_threshold=10,
            remove_overlay_elements=True,
        )
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            result = await crawler.arun(url=url, config=run_cfg)
            if result.success:
                # Use fit_markdown if available (cleaner), fallback to markdown
                md = getattr(result, "fit_markdown", None) or result.markdown or ""
                title = ""
                if result.metadata:
                    title = result.metadata.get("title", url)
                return {
                    "url": url,
                    "markdown": md,
                    "title": title or url,
                    "links": _collect_links(result, url),
                    "success": True,
                }
            else:
                err = getattr(result, "error_message", "unknown error")
                return {"url": url, "markdown": "", "title": url, "links": [], "success": False, "error": str(err)}
    except ImportError:
        return await _fallback_crawl(url)
    except Exception as e:
        return {"url": url, "markdown": "", "title": url, "links": [], "success": False, "error": str(e)}


async def crawl_site(base_url: str, max_pages: int = 10) -> List[Dict[str, Any]]:
    """
    Spider a site starting from base_url, following internal links.
    Returns list of crawled page results.
    """
    visited: set = set()
    to_visit: List[str] = [base_url]
    results: List[Dict[str, Any]] = []
    base_domain = urlparse(base_url).netloc

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        result = await crawl_url(url)
        results.append(result)

        # Extract links from crawl result
        if result.get("success"):
            for link in result.get("links", []):
                if urlparse(link).netloc == base_domain and link not in visited:
                    to_visit.append(link)
            # Fallback: extract from markdown
            if not result.get("links"):
                md_links = _extract_links_from_markdown(
                    result.get("markdown", ""), base_url, base_domain
                )
                for link in md_links:
                    if link not in visited:
                        to_visit.append(link)

    return results


def _collect_links(result: Any, base_url: str) -> List[str]:
    """Collect internal links from a Crawl4AI result object."""
    links = []
    try:
        # Crawl4AI exposes links as result.links dict
        if hasattr(result, "links") and isinstance(result.links, dict):
            for item in result.links.get("internal", []):
                href = item.get("href", "") if isinstance(item, dict) else str(item)
                if href and href.startswith("http"):
                    links.append(href)
                elif href and href.startswith("/"):
                    links.append(urljoin(base_url, href))
    except Exception:
        pass
    return links[:20]


def _extract_links_from_markdown(markdown: str, base_url: str, base_domain: str) -> List[str]:
    """Extract internal links from markdown text as fallback."""
    import re
    links = []
    patterns = [
        r'\[.*?\]\((https?://[^\)]+)\)',
        r'href=["\'](https?://[^"\']+)["\']',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, markdown):
            url = match.group(1)
            if urlparse(url).netloc == base_domain:
                links.append(url)
            elif url.startswith("/"):
                links.append(urljoin(base_url, url))
    return links[:20]


async def _fallback_crawl(url: str) -> Dict[str, Any]:
    """Basic HTTP fallback without JS rendering."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        import re
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return {"url": url, "markdown": text[:8000], "title": url, "links": [], "success": True}
    except Exception as e:
        return {"url": url, "markdown": "", "title": url, "links": [], "success": False, "error": str(e)}
