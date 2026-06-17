"""
cache/kv_cache.py
──────────────────
Manages Anthropic prompt caching (cache_control) for token savings.

Strategy:
  - The person's full bio/resume text is the STABLE PREFIX — cached on first call
  - Each QA call reuses the cached prefix → ~90% input token savings
  - cache_control type="ephemeral" lasts ~5 minutes (sufficient for a session)
  - For claude-sonnet-4-x: prompt caching is built-in, no beta flag needed.
    Just include cache_control blocks in your system prompt content list.

Usage stats are tracked in-memory and exposed to the frontend.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class CacheStats:
    total_calls: int = 0
    cache_hits: int = 0
    input_tokens_total: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens_total: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return round(self.cache_hits / self.total_calls, 3)

    @property
    def tokens_saved(self) -> int:
        """Cache reads are charged at ~10% of normal price — report as saved."""
        return self.cache_read_tokens

    @property
    def net_savings_pct(self) -> float:
        """Percentage of input tokens served from cache."""
        if self.input_tokens_total == 0:
            return 0.0
        return round(self.cache_read_tokens / max(self.input_tokens_total, 1) * 100, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "hit_rate_pct": round(self.hit_rate * 100, 1),
            "input_tokens_total": self.input_tokens_total,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens_total": self.output_tokens_total,
            "tokens_saved": self.tokens_saved,
            "net_savings_pct": self.net_savings_pct,
        }


_stats = CacheStats()


def get_cache_stats() -> CacheStats:
    return _stats


def build_system_prompt_with_cache(bio_text: str, person_name: str) -> List[Dict[str, Any]]:
    """
    Build the system prompt blocks with cache_control on the stable bio prefix.

    Block 1 (CACHED): Person bio + core knowledge — stable across all QA calls.
      cache_control marks this for server-side caching. After the first call,
      Anthropic serves this block from cache at ~10% of the normal token cost.

    Block 2 (not cached): Runtime instruction guidelines — can vary per call.

    Note: No betas= flag needed for claude-sonnet-4-x.
      Include cache_control directly in the content list.
    """
    bio_block = {
        "text": f"""You are an expert knowledge assistant for {person_name}'s professional profile.

You have deep knowledge of {person_name}'s background, experience, skills, and projects based on 
the following comprehensive profile information:

═══════════════════════════════════════════════
KNOWLEDGE BASE — {person_name.upper()}
═══════════════════════════════════════════════
{bio_text}
═══════════════════════════════════════════════

Answer questions about {person_name} accurately and concisely.
Always cite your sources using [source §section] notation.
If you're not confident about something, say so clearly.
Do not invent information not present in the knowledge base.""",
    }

    instruction_block = {
        "type": "text",
        "text": """Response guidelines:
- Be an intelligent, conversational assistant. DO NOT just quote lines.
- Synthesize information across timelines, roles, GitHub repositories, and LinkedIn data to provide smart correlations.
- If asked about a role or skill, provide context from multiple sources if possible (e.g., connect a project at ISRO to a GitHub repo or a skill listed elsewhere).
- Cite sources inline using [source §section] notation naturally within your narrative (e.g., "While at ISRO [resume §2], Devshree did X, which aligns with her GitHub project [github §3]").
- If asked about something not in the knowledge base, say "I don't have that information"
- Keep answers insightful and complete, highlighting career progression or skill overlap.
- When graph context is provided, use it to add structural relationships.""",
    }

    return [bio_block, instruction_block]


def estimate_token_savings(bio_text: str) -> Dict[str, int]:
    """
    Rough estimate of tokens saved per call after first cache write.
    Useful for frontend display before any calls are made.
    """
    # ~4 chars per token rough estimate
    bio_tokens = len(bio_text) // 4
    instructions_tokens = 100
    total_system_tokens = bio_tokens + instructions_tokens
    # After first call, bio block served at ~10% cost
    savings_per_call = int(bio_tokens * 0.90)
    return {
        "estimated_bio_tokens": bio_tokens,
        "estimated_savings_per_call": savings_per_call,
        "savings_pct": 90,
    }


def record_usage(usage_obj: Any) -> None:
    """Record token usage from a Gemini API response."""
    _stats.total_calls += 1
    if not usage_obj:
        return
    if hasattr(usage_obj, "prompt_token_count"):
        _stats.input_tokens_total += usage_obj.prompt_token_count or 0
    if hasattr(usage_obj, "cached_content_token_count"):
        cache_read = usage_obj.cached_content_token_count or 0
        _stats.cache_read_tokens += cache_read
        if cache_read > 0:
            _stats.cache_hits += 1
    if hasattr(usage_obj, "candidates_token_count"):
        _stats.output_tokens_total += usage_obj.candidates_token_count or 0

