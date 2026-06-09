"""
Token + cost accounting for LLM calls.

Vertex passes through the Anthropic per-model rates, so we price each response's
`usage` block and accumulate a process-wide total. Call record() right after a
messages.create() and the cost lands in the logs; the running total is exposed
on /health for a real cost-per-assessment read instead of estimates.
"""
import logging
import threading

logger = logging.getLogger(__name__)

# USD per 1,000,000 tokens.
PRICES = {
    "sonnet": {"in": 3.00, "out": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "haiku":  {"in": 1.00, "out": 5.00,  "cache_read": 0.10, "cache_write": 1.25},
    "opus":   {"in": 5.00, "out": 25.00, "cache_read": 0.50, "cache_write": 6.25},
}


def _tier(model: str) -> str:
    m = (model or "").lower()
    if "haiku" in m:
        return "haiku"
    if "opus" in m:
        return "opus"
    return "sonnet"


_lock = threading.Lock()
_totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
           "cost_usd": 0.0, "calls": 0}


def record(model: str, usage, tag: str = "") -> float:
    """Log and accumulate the cost of one response. Never raises."""
    try:
        in_t = int(getattr(usage, "input_tokens", 0) or 0)
        out_t = int(getattr(usage, "output_tokens", 0) or 0)
        cr = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cw = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    except Exception:
        return 0.0
    p = PRICES[_tier(model)]
    cost = (in_t * p["in"] + out_t * p["out"]
            + cr * p["cache_read"] + cw * p["cache_write"]) / 1_000_000
    with _lock:
        _totals["input"] += in_t
        _totals["output"] += out_t
        _totals["cache_read"] += cr
        _totals["cache_write"] += cw
        _totals["cost_usd"] += cost
        _totals["calls"] += 1
    logger.info(f"[cost] {model} in={in_t} out={out_t} "
                f"cache_r={cr} cache_w={cw} ${cost:.4f} {tag}".rstrip())
    return cost


def totals() -> dict:
    with _lock:
        t = dict(_totals)
    t["cost_usd"] = round(t["cost_usd"], 4)
    return t
