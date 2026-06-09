"""
Shared 429 / RESOURCE_EXHAUSTED backoff for Vertex calls.

Vertex enforces a per-minute tokens-per-minute quota per base model. When it is
exhausted, messages.create() raises a 429. The SDK's own retries use short
backoff; a per-minute bucket needs a longer wait. acall() runs a synchronous
create in the executor and retries on rate-limit errors with a backoff long
enough to outlast the bucket, honouring Retry-After when present. Non-rate-limit
errors surface immediately.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)

# Seconds to wait between attempts (≈46s total) — enough to let a per-minute
# tokens-per-minute bucket refill, without holding a request open forever.
RATE_LIMIT_BACKOFF = [4, 12, 30]


def is_rate_limited(err: Exception) -> bool:
    m = str(err)
    return ("429" in m or "RESOURCE_EXHAUSTED" in m
            or ("rate" in m.lower() and "limit" in m.lower()))


async def acall(sync_fn, label: str = ""):
    """Run sync_fn() in the default executor, retrying on quota/429 errors."""
    loop = asyncio.get_event_loop()
    last = None
    for attempt in range(len(RATE_LIMIT_BACKOFF) + 1):
        try:
            return await loop.run_in_executor(None, sync_fn)
        except Exception as e:
            if attempt < len(RATE_LIMIT_BACKOFF) and is_rate_limited(e):
                wait = RATE_LIMIT_BACKOFF[attempt]
                hdrs = getattr(getattr(e, "response", None), "headers", None)
                if hdrs is not None:
                    try:
                        wait = max(wait, float(hdrs.get("retry-after")))
                    except (TypeError, ValueError):
                        pass
                logger.warning(
                    f"Vertex quota/429{(' ' + label) if label else ''}; retry "
                    f"{attempt + 1}/{len(RATE_LIMIT_BACKOFF)} in {wait:.0f}s")
                await asyncio.sleep(wait)
                last = e
                continue
            raise
    raise last
