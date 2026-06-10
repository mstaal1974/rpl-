"""
Database layer — Firestore + BigQuery + in-memory fallback.
v4: adds assessment creation, invite tokens, and step-by-step progress persistence.
"""
import os, json, uuid, logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Read project lazily — env vars aren't injected at module load time on Cloud Run
FIRESTORE_DB    = os.getenv("FIRESTORE_DATABASE", "(default)")
BQ_DATASET      = os.getenv("BQ_DATASET", "rpl_audit")
BQ_TABLE        = os.getenv("BQ_TABLE", "assessment_records")
RETENTION_YEARS = int(os.getenv("ASQA_RETENTION_YEARS", "2"))

def _gcp_project():
    """Resolve GCP project ID from any of the env vars Cloud Run sets.
    Returns None if unset — the Google client libraries then infer the project
    from Application Default Credentials (the correct behaviour on Cloud Run).
    No project ID is hard-coded in source."""
    return (os.getenv("GOOGLE_CLOUD_PROJECT") or
            os.getenv("ANTHROPIC_VERTEX_PROJECT_ID") or
            os.getenv("GCLOUD_PROJECT") or
            None)

_fs = None
_bq = None
_store: dict = {}       # in-memory fallback for local dev
_token_index: dict = {} # token → assessment_id cache


def _firestore():
    global _fs
    if _fs is None:
        gcp = _gcp_project()
        try:
            from google.cloud import firestore
            _fs = firestore.AsyncClient(project=gcp, database=FIRESTORE_DB)
        except Exception as e:
            logger.warning(f"Firestore unavailable: {e} — using in-memory store")
    return _fs


def _bigquery():
    global _bq
    if _bq is None:
        gcp = _gcp_project()
        try:
            from google.cloud import bigquery
            _bq = bigquery.Client(project=gcp)
        except Exception as e:
            logger.warning(f"BigQuery unavailable: {e}")
    return _bq


# ── Assessment creation ───────────────────────────────────────────────────────

async def create_assessment(trainer_id: str, data: dict) -> str:
    """
    Create a new assessment record. Returns (assessment_id, invite_token).
    data must include: unit_codes (list), candidate (dict), trainer_name,
                       org_id, trainer_user_id   (multi-tenant fields)
    """
    assessment_id = str(uuid.uuid4())
    invite_token  = str(uuid.uuid4()).replace("-", "")[:24]

    org_id          = data.get("org_id", "")
    trainer_user_id = data.get("trainer_user_id", "")

    record = {
        "assessment_id":  assessment_id,
        "invite_token":   invite_token,
        "org_id":         org_id,             # tenant scope
        "trainer_user_id": trainer_user_id,   # owning user (login id)
        "trainer_id":     trainer_id,
        "trainer_name":   data.get("trainer_name", ""),
        "trainer_email":  data.get("trainer_email", ""),
        "unit_codes":     data.get("unit_codes", []),
        "candidate":      data.get("candidate", {}),
        "notes":          data.get("notes", ""),
        "status":         "INVITED",      # INVITED → IN_PROGRESS → SUBMITTED → COMPLETE
        "created_at":     datetime.now(timezone.utc).isoformat(),
        "submitted_at":   None,
        "completed_at":   None,
        "invite_sent_at": None,
        "progress": {
            "current_step":       "welcome",
            "completed_steps":    [],
            "checklist":          {},
            "knowledge_responses":{},
            "knowledge_analyses": {},
            "uploads":            {},
            "candidate_notes":    {},
            "mapping":            None,
            "gap_analysis":       None,
            "conversation_records": [],
        },
        "_meta": {"org_id": org_id,
                  "privacy_policy": "Privacy Act 1988 APP compliant",
                  "asqa_retention_years": RETENTION_YEARS}
    }

    db = _firestore()
    firestore_ok = False
    if db:
        try:
            await db.collection("rpl_assessments").document(assessment_id).set(record)
            await db.collection("rpl_token_index").document(invite_token).set({
                "assessment_id": assessment_id,
                "created_at":    record["created_at"],
            })
            firestore_ok = True
            logger.info(f"Assessment {assessment_id} written to Firestore OK")
            await _mirror_bq(assessment_id, "created", record)
        except Exception as e:
            logger.error(f"Firestore create FAILED — assessment {assessment_id} "
                         f"in memory only (will not survive redeploy): {e}")
            # Attempt emergency retry once
            try:
                import asyncio
                await asyncio.sleep(1)
                await db.collection("rpl_assessments").document(assessment_id).set(record)
                await db.collection("rpl_token_index").document(invite_token).set({
                    "assessment_id": assessment_id,
                    "created_at":    record["created_at"],
                })
                firestore_ok = True
                logger.info(f"Firestore write succeeded on retry for {assessment_id}")
            except Exception as e2:
                logger.error(f"Firestore retry also failed: {e2}")
    else:
        logger.warning("Firestore not configured — assessment in memory only")

    _store[assessment_id] = record
    _token_index[invite_token] = assessment_id
    record["_firestore_ok"] = firestore_ok
    return assessment_id, invite_token


# ── Get by invite token ───────────────────────────────────────────────────────

async def get_by_token(token: str) -> Optional[dict]:
    """Retrieve assessment by invite token.
    Uses token_index collection for O(1) lookup — no composite index needed.
    Falls back to query scan for legacy assessments created before index existed.
    """
    # Check in-memory index first (fast path)
    if token in _token_index:
        rec = await get_assessment(_token_index[token])
        if rec:
            return rec

    db = _firestore()
    if db:
        # Path 1: token_index lookup (O(1), no index required)
        try:
            idx_doc = await db.collection("rpl_token_index").document(token).get()
            if idx_doc.exists:
                assessment_id = idx_doc.to_dict().get("assessment_id")
                if assessment_id:
                    _token_index[token] = assessment_id  # cache
                    rec = await get_assessment(assessment_id)
                    if rec:
                        return rec
        except Exception as e:
            logger.warning(f"Token index lookup failed: {e}")

        # Path 2: Legacy — scan rpl_assessments where invite_token matches
        try:
            col = db.collection("rpl_assessments").where("invite_token", "==", token)
            async for doc in col.stream():
                data = doc.to_dict()
                if data:
                    # Backfill the token index so future lookups are fast
                    try:
                        await db.collection("rpl_token_index").document(token).set({
                            "assessment_id": data["assessment_id"],
                            "created_at":    data.get("created_at", ""),
                        })
                        _token_index[token] = data["assessment_id"]
                    except Exception:
                        pass
                    return data
        except Exception as e:
            logger.error(f"Token query scan failed: {e}")

        # Path 3: Full collection scan (last resort for very old records)
        try:
            async for doc in db.collection("rpl_assessments").stream():
                data = doc.to_dict()
                if data and data.get("invite_token") == token:
                    logger.info(f"Token found via full scan — backfilling index for {token[:8]}")
                    try:
                        await db.collection("rpl_token_index").document(token).set({
                            "assessment_id": data["assessment_id"],
                            "created_at":    data.get("created_at", ""),
                        })
                        _token_index[token] = data["assessment_id"]
                    except Exception:
                        pass
                    return data
        except Exception as e:
            logger.error(f"Full scan failed: {e}")

    # In-memory fallback
    for r in _store.values():
        if r.get("invite_token") == token:
            return r
    return None


# ── Progress save / load ──────────────────────────────────────────────────────

async def save_progress(assessment_id: str, progress: dict) -> bool:
    """
    Save the student's current progress. Called automatically on every step completion.
    progress dict merges into the existing progress field.
    """
    db = _firestore()
    if db:
        try:
            doc = db.collection("rpl_assessments").document(assessment_id)
            await doc.update({"progress": progress,
                              "status": "IN_PROGRESS",
                              "last_updated": datetime.now(timezone.utc).isoformat()})
            return True
        except Exception as e:
            logger.error(f"Progress save error: {e}")

    if assessment_id in _store:
        _store[assessment_id]["progress"] = progress
        _store[assessment_id]["status"] = "IN_PROGRESS"
    return True


async def set_status(assessment_id: str, status: str) -> bool:
    """Set the assessment status directly (used to preserve SUBMITTED/COMPLETE
    when an assessor edits progress, e.g. adds a competency-conversation record)."""
    db = _firestore()
    if db:
        try:
            await db.collection("rpl_assessments").document(assessment_id).update(
                {"status": status})
            return True
        except Exception as e:
            logger.error(f"Status update error: {e}")
    if assessment_id in _store:
        _store[assessment_id]["status"] = status
    return True


async def load_progress(assessment_id: str) -> Optional[dict]:
    """Load the latest progress for an assessment."""
    rec = await get_assessment(assessment_id)
    if rec:
        return rec.get("progress", {})
    return None


# ── Status updates ────────────────────────────────────────────────────────────

async def submit_assessment(assessment_id: str) -> bool:
    """Mark assessment as submitted by student."""
    db = _firestore()
    now = datetime.now(timezone.utc).isoformat()
    if db:
        try:
            await db.collection("rpl_assessments").document(assessment_id).update({
                "status": "SUBMITTED", "submitted_at": now})
            await _mirror_bq(assessment_id, "submitted", {"submitted_at": now})
            return True
        except Exception as e:
            logger.error(f"Submit error: {e}")

    if assessment_id in _store:
        _store[assessment_id]["status"] = "SUBMITTED"
        _store[assessment_id]["submitted_at"] = now
    return True


async def complete_assessment(assessment_id: str) -> bool:
    """Mark assessment as complete by trainer."""
    db = _firestore()
    now = datetime.now(timezone.utc).isoformat()
    expiry = (datetime.now(timezone.utc) + timedelta(days=365*RETENTION_YEARS)).isoformat()
    if db:
        try:
            await db.collection("rpl_assessments").document(assessment_id).update({
                "status": "COMPLETE", "completed_at": now,
                "retention_expiry": expiry})
            return True
        except Exception as e:
            logger.error(f"Complete error: {e}")

    if assessment_id in _store:
        _store[assessment_id]["status"] = "COMPLETE"
    return True


# ── Standard CRUD ─────────────────────────────────────────────────────────────

async def save_assessment(assessment_id: str, record_type: str, data: dict) -> bool:
    payload = {**data, "_meta": {"record_type": record_type,
                                  "assessment_id": assessment_id,
                                  "saved_at": datetime.now(timezone.utc).isoformat()}}
    db = _firestore()
    if db:
        try:
            doc = db.collection("rpl_assessments").document(assessment_id)
            await doc.collection("records").document(record_type).set(payload, merge=True)
            await _mirror_bq(assessment_id, record_type, payload)
            return True
        except Exception as e:
            logger.error(f"Firestore save error: {e}")

    if assessment_id not in _store:
        _store[assessment_id] = {}
    _store[assessment_id][record_type] = payload
    return True


async def get_assessment(assessment_id: str) -> Optional[dict]:
    db = _firestore()
    if db:
        try:
            doc = await db.collection("rpl_assessments").document(assessment_id).get()
            if doc.exists:
                return doc.to_dict()
        except Exception as e:
            logger.error(f"Firestore get error: {e}")

    return _store.get(assessment_id)


async def list_assessments(trainer_id: Optional[str] = None,
                           status: Optional[str] = None,
                           org_id: Optional[str] = None,
                           trainer_user_id: Optional[str] = None,
                           limit: int = 200) -> list:
    """
    List assessments with optional filters.
    org_id  — restrict to a single tenant (REQUIRED in multi-tenant calls)
    trainer_user_id — restrict to one trainer's own records (trainers only see their own)
    """
    db = _firestore()
    if db:
        try:
            col = db.collection("rpl_assessments")
            if org_id:
                col = col.where("org_id", "==", org_id)
            if trainer_user_id:
                col = col.where("trainer_user_id", "==", trainer_user_id)
            if trainer_id:
                col = col.where("trainer_id", "==", trainer_id)
            if status:
                col = col.where("status", "==", status)
            results = []
            async for doc in col.limit(limit).stream():
                results.append(doc.to_dict())
            return results
        except Exception as e:
            logger.error(f"Firestore list error: {e}")

    results = list(_store.values())
    if org_id:
        results = [r for r in results if r.get("org_id") == org_id]
    if trainer_user_id:
        results = [r for r in results if r.get("trainer_user_id") == trainer_user_id]
    if trainer_id:
        results = [r for r in results if r.get("trainer_id") == trainer_id]
    if status:
        results = [r for r in results if r.get("status") == status]
    return results[:limit]


async def _mirror_bq(assessment_id: str, record_type: str, data: dict):
    import asyncio
    bq = _bigquery()
    if not bq:
        return

    def _insert():
        # Use the project the BigQuery client actually resolved (handles the
        # case where the project is inferred from ADC rather than an env var).
        table = f"{bq.project}.{BQ_DATASET}.{BQ_TABLE}"
        rows = [{"assessment_id": assessment_id, "record_type": record_type,
                 "data_json": json.dumps(data),
                 "inserted_at": datetime.now(timezone.utc).isoformat(),
                 "org_id": data.get("org_id", "") if isinstance(data, dict) else "",
                 "retention_years": RETENTION_YEARS}]
        errors = bq.insert_rows_json(table, rows)
        if errors:
            logger.warning(f"BigQuery errors: {errors}")

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _insert)
    except Exception as e:
        logger.warning(f"BigQuery mirror failed: {e}")
