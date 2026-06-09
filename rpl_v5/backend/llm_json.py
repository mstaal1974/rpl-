"""
Tolerant JSON extraction from LLM replies.

Model output is asked to be "ONLY valid JSON", but in practice it can arrive
wrapped in ```json fences, with a prose preamble, or with trailing commentary.
The old pattern — `text.replace("```json","").replace("```","")` then
`json.loads` — fails on any of those and 500s the endpoint. This extractor
strips fences, tries a direct parse, and falls back to the outermost {...}/[...]
span, so a stray sentence around the JSON no longer breaks the request.
"""
import json
import re


def extract_json(raw):
    """Parse JSON from a model reply, tolerating code fences and surrounding prose."""
    if isinstance(raw, (dict, list)):
        return raw
    s = (raw or "").strip()
    # Strip a leading ```json / ``` fence and a trailing ``` fence.
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        # Fall back to the outermost object or array span.
        candidates = []
        for open_c, close_c in (("{", "}"), ("[", "]")):
            i, j = s.find(open_c), s.rfind(close_c)
            if i != -1 and j != -1 and j > i:
                candidates.append((i, s[i:j + 1]))
        # Prefer whichever bracket type appears first in the text.
        for _, span in sorted(candidates):
            try:
                return json.loads(span)
            except Exception:
                continue
        raise
