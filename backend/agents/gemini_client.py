"""
backend/agents/gemini_client.py
────────────────────────────────
Resilient Gemini API client with:
  • Exponential backoff retry on 429 RESOURCE_EXHAUSTED
  • Model fallback chain (primary → fallbacks)
  • Structured error classification
  • Centralised logging

Usage:
    from backend.agents.gemini_client import gemini_generate, gemini_stream

    # Non-streaming
    text = gemini_generate(prompt, config=...)

    # Streaming (async generator of text chunks)
    async for chunk in gemini_stream(prompt):
        ...
"""
import asyncio
import logging
import os
import time
from typing import Any, AsyncGenerator, Iterator, Optional

from google import genai
from google.genai import types

from backend.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger("gemini_client")
logging.basicConfig(level=logging.INFO)

# ── Model fallback chain ──────────────────────────────────────────────────────
# Verified available models for this API key (v1beta endpoint).
# Primary is gemini-2.0-flash (higher RPM than 2.5-flash on free tier).
# Fallbacks go lighter first, then step up to 2.5-flash only as last resort.
_FALLBACK_MODELS = [
    "gemini-2.0-flash-lite",   # lightest, highest quota
    "gemini-2.5-flash-lite",   # mid-tier
    "gemini-2.5-flash",        # original primary, use as last fallback
]

# ── Retry settings ────────────────────────────────────────────────────────────
_MAX_RETRIES   = 3          # attempts per model
_BASE_DELAY    = 2.0        # seconds before first retry
_MAX_DELAY     = 30.0       # cap on exponential backoff
_BACKOFF_MULT  = 2.5        # multiplier per retry

# ── User-facing error messages ────────────────────────────────────────────────
MSG_QUOTA_EXHAUSTED = (
    "The AI service is temporarily busy. Please try again in a few moments."
)
MSG_API_KEY_MISSING = (
    "The AI service is not configured. Please set GEMINI_API_KEY in your .env file."
)
MSG_GENERIC_ERROR   = (
    "Something went wrong with the AI service. Please try again."
)


# ── Error classification ──────────────────────────────────────────────────────

class GeminiQuotaError(Exception):
    """Raised when all models + retries are exhausted due to rate limiting."""

class GeminiAPIKeyError(Exception):
    """Raised when the API key is missing or invalid."""

class GeminiFatalError(Exception):
    """Raised for non-retryable errors (bad request, etc.)."""


def _is_quota_error(exc: Exception) -> bool:
    """Return True for 429 RESOURCE_EXHAUSTED / rate limit errors."""
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg or "rate_limit" in msg

def _is_key_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "api_key" in msg or "invalid_api_key" in msg or "401" in msg or "403" in msg

def _is_model_not_found(exc: Exception) -> bool:
    """Return True for 404 NOT_FOUND — model name unavailable on this endpoint."""
    msg = str(exc).lower()
    return "404" in msg or "not_found" in msg or "not found" in msg

def _is_retryable(exc: Exception) -> bool:
    """Only quota/rate-limit errors are worth retrying on the same model."""
    return _is_quota_error(exc)


# ── Client factory (cached) ───────────────────────────────────────────────────

_client: Optional[genai.Client] = None

def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise GeminiAPIKeyError("GEMINI_API_KEY is not set. Add it to your .env file.")
        logger.info("[GeminiClient] Initialised with model primary=%s", GEMINI_MODEL)
        _client = genai.Client(api_key=api_key)
    return _client


# ── Non-streaming with retry + fallback ──────────────────────────────────────

def gemini_generate(
    prompt: str,
    config: Optional[types.GenerateContentConfig] = None,
) -> str:
    """
    Generate content synchronously.
    Retries on 429 with backoff, then tries fallback models.
    Raises GeminiQuotaError if all attempts fail.
    """
    client = _get_client()
    model_chain = [GEMINI_MODEL] + _FALLBACK_MODELS

    last_exc: Exception = RuntimeError("No attempts made")

    for model in model_chain:
        delay = _BASE_DELAY
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info("[GeminiClient] generate model=%s attempt=%d", model, attempt)
                kwargs: dict[str, Any] = {"model": model, "contents": prompt}
                if config:
                    kwargs["config"] = config
                resp = client.models.generate_content(**kwargs)
                logger.info("[GeminiClient] Success model=%s", model)
                return resp.text or ""
            except Exception as exc:
                last_exc = exc
                if _is_key_error(exc):
                    logger.error("[GeminiClient] API key error: %s", exc)
                    raise GeminiAPIKeyError(str(exc)) from exc
                if _is_model_not_found(exc):
                    logger.warning("[GeminiClient] Model not found: %s — skipping to next", model)
                    break  # skip to next model immediately
                if _is_quota_error(exc):
                    if attempt < _MAX_RETRIES:
                        logger.warning(
                            "[GeminiClient] Quota hit model=%s attempt=%d — retrying in %.1fs",
                            model, attempt, delay,
                        )
                        time.sleep(delay)
                        delay = min(delay * _BACKOFF_MULT, _MAX_DELAY)
                    else:
                        logger.warning("[GeminiClient] Quota exhausted for model=%s — trying next", model)
                    continue
                # Other non-retryable error — still try next model
                logger.warning("[GeminiClient] Error model=%s: %s — trying next model", model, exc)
                break  # try next model

    logger.error("[GeminiClient] All models failed. Last error: %s", last_exc)
    raise GeminiQuotaError(MSG_QUOTA_EXHAUSTED) from last_exc


# ── Streaming with retry + fallback ──────────────────────────────────────────

def gemini_stream(
    prompt: str,
    config: Optional[types.GenerateContentConfig] = None,
) -> Iterator[str]:
    """
    Stream content synchronously (sync generator of text chunks).
    On quota error: tries retry + fallback models, then yields the user-friendly
    error message as a single chunk.
    """
    client = _get_client()
    model_chain = [GEMINI_MODEL] + _FALLBACK_MODELS

    for model in model_chain:
        delay = _BASE_DELAY
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info("[GeminiClient] stream model=%s attempt=%d", model, attempt)
                kwargs: dict[str, Any] = {"model": model, "contents": prompt}
                if config:
                    kwargs["config"] = config
                chunks = client.models.generate_content_stream(**kwargs)
                for chunk in chunks:
                    if chunk.text:
                        yield chunk.text
                logger.info("[GeminiClient] Stream complete model=%s", model)
                return  # success
            except Exception as exc:
                if _is_key_error(exc):
                    logger.error("[GeminiClient] API key error: %s", exc)
                    yield MSG_API_KEY_MISSING
                    return
                if _is_model_not_found(exc):
                    logger.warning("[GeminiClient] Stream model not found: %s — skipping to next", model)
                    break  # try next model
                if _is_quota_error(exc):
                    if attempt < _MAX_RETRIES:
                        logger.warning(
                            "[GeminiClient] Quota hit stream model=%s attempt=%d — retrying in %.1fs",
                            model, attempt, delay,
                        )
                        time.sleep(delay)
                        delay = min(delay * _BACKOFF_MULT, _MAX_DELAY)
                        continue
                    else:
                        logger.warning("[GeminiClient] Quota exhausted stream model=%s — trying next", model)
                    break
                # Other error — try next model
                logger.warning("[GeminiClient] Stream error model=%s: %s — trying next model", model, exc)
                break

    logger.error("[GeminiClient] All models quota-exhausted for streaming.")
    yield MSG_QUOTA_EXHAUSTED


# ── Async wrappers (for async FastAPI routes) ─────────────────────────────────

async def gemini_generate_async(
    prompt: str,
    config: Optional[types.GenerateContentConfig] = None,
) -> str:
    """Async wrapper around gemini_generate — runs in thread pool to avoid blocking."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: gemini_generate(prompt, config))


async def gemini_stream_async(
    prompt: str,
    config: Optional[types.GenerateContentConfig] = None,
) -> AsyncGenerator[str, None]:
    """
    Async wrapper around gemini_stream.
    Yields text chunks; on quota error yields the friendly message.
    """
    loop = asyncio.get_event_loop()
    import concurrent.futures

    def _iter():
        return list(gemini_stream(prompt, config))

    with concurrent.futures.ThreadPoolExecutor() as pool:
        chunks = await loop.run_in_executor(pool, _iter)

    for chunk in chunks:
        yield chunk


def is_quota_error_message(text: str) -> bool:
    """Check if a text string is one of our quota error messages."""
    return text in (MSG_QUOTA_EXHAUSTED, MSG_API_KEY_MISSING, MSG_GENERIC_ERROR)
