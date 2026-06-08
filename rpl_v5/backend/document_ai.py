"""
Google Document AI integration.
Falls back to basic text extraction if GCP not configured.
"""
import os, re, asyncio, logging
from typing import Optional

logger = logging.getLogger(__name__)

GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("DOCUMENT_AI_LOCATION", "us")
PROCESSOR_ID = os.getenv("DOCAI_DEFAULT_PROCESSOR_ID")


async def parse_document(content: bytes, mime_type: str, doc_type: str = "default") -> dict:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _parse_sync, content, mime_type, doc_type)
    return result


def _parse_sync(content: bytes, mime_type: str, doc_type: str) -> dict:
    if not GCP_PROJECT or not PROCESSOR_ID:
        logger.info("Document AI not configured — returning basic text extraction")
        text = content.decode("utf-8", errors="replace") if mime_type in [
            "text/plain", "text/html"] else f"[Binary file — {len(content)} bytes]"
        return {"raw_text": text, "entities": [], "token_count": len(text.split()),
                "confidence": 0.75, "pages": 1, "mime_type": mime_type}

    try:
        from google.cloud import documentai_v1 as documentai
        from google.api_core.client_options import ClientOptions

        opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
        client = documentai.DocumentProcessorServiceClient(client_options=opts)
        name = f"projects/{GCP_PROJECT}/locations/{LOCATION}/processors/{PROCESSOR_ID}"
        raw_doc = documentai.RawDocument(content=content, mime_type=mime_type)
        request = documentai.ProcessRequest(name=name, raw_document=raw_doc)
        response = client.process_document(request=request)
        document = response.document

        entities = [{"type": e.type_, "mention": e.mention_text,
                     "confidence": round(e.confidence, 3)}
                    for e in document.entities]

        return {"raw_text": document.text or "", "entities": entities,
                "token_count": len((document.text or "").split()),
                "confidence": 0.90, "pages": len(document.pages), "mime_type": mime_type}

    except Exception as e:
        logger.error(f"Document AI error: {e}")
        return {"raw_text": "", "entities": [], "token_count": 0,
                "confidence": 0.0, "pages": 0, "mime_type": mime_type, "error": str(e)}


def redact_sensitive(parsed: dict) -> dict:
    """Remove government identifiers — Privacy Act 1988 APP 9 compliance."""
    text = parsed.get("raw_text", "")
    patterns = [
        (r"\b\d{3}\s?\d{3}\s?\d{3}\b", "[TFN REDACTED]"),
        (r"\b\d{4}\s?\d{5}\s?\d{1}\b", "[MEDICARE REDACTED]"),
        (r"\b[A-Z]{1,2}\d{7}\b", "[PASSPORT REDACTED]"),
        (r"\b\d{2,3}\s?\d{3}\s?\d{3}\s?\d{3}\b", "[ACN/ABN REDACTED]"),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    parsed["raw_text"] = text
    return parsed
