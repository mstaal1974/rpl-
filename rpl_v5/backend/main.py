"""
RPL System v5.0 — Multi-tenant
Organisation (1) ─< Users (admin = Administration & Compliance, trainer = Trainers)
Each user has their own login. All data is scoped to org_id.
"""
import os, json, base64, logging, secrets, asyncio
from datetime import datetime, timezone
from typing import Optional
import anthropic
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from .unit_registry import registry, UnitOfCompetency, import_from_tgau
from .mapping_engine import (run_mapping, run_gap_analysis, analyse_knowledge_response,
    run_cross_unit_mapping, generate_third_party_report_template,
    generate_evidence_portfolio_summary, generate_benchmark_gap_report,
    generate_assessment_patterns, generate_industry_context_profile,
    generate_determination_worksheet, run_pre_assessment_screen,
    generate_knowledge_questions_for_unit, evaluate_knowledge_answer_detailed)
from .orchestrator import (orchestrate_rpl_assessment, orchestrate_multi_unit_assessment)
from .mapping_engine import detect_ai_usage, analyse_assessment_for_ai_usage
from .database import (
    create_assessment, get_by_token, save_progress, load_progress,
    submit_assessment, complete_assessment,
    save_assessment, get_assessment, list_assessments
)
from . import auth as _auth
from .auth import (
    current_user, require_admin, require_superadmin, require_trainer_or_admin,
    authenticate, issue_token,
    create_user as auth_create_user,
    list_users as auth_list_users,
    update_user as auth_update_user,
    delete_user as auth_delete_user,
    list_orgs as auth_list_orgs,
    create_org as auth_create_org,
    get_org as auth_get_org,
    update_org_settings as auth_update_org,
    bootstrap_first_org,
    stats as auth_stats,
    create_org_with_admin, list_orgs_with_counts, set_org_active,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RPL System (multi-tenant)", version="5.0.0")

# ── Background question enrichment queue ──────────────────────────────────────
# Units uploaded via Excel get template questions immediately.
# This queue enriches them with AI-quality questions asynchronously.
_enrich_queue: asyncio.Queue = None

async def _enrich_worker():
    """Background worker — upgrades template questions to AI-quality for queued units."""
    global _enrich_queue
    logger.info("Question enrichment worker started")
    while True:
        try:
            item = await asyncio.wait_for(_enrich_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            continue
        except Exception:
            await asyncio.sleep(1)
            continue

        unit_code      = item.get("unit_code")
        industry_context = item.get("industry_context", "")
        try:
            unit = registry.get(unit_code)
            if not unit:
                logger.warning(f"Enrich: unit {unit_code} not found")
                _enrich_queue.task_done()
                continue

            # Check if already has AI-quality questions
            if (unit.knowledge_questions and
                unit.knowledge_questions[0].model_answer_guide.expected_knowledge_points):
                logger.debug(f"Enrich: {unit_code} already has AI questions — skipping")
                _enrich_queue.task_done()
                continue

            logger.info(f"Enriching questions for {unit_code}...")
            questions = await generate_knowledge_questions_for_unit(
                get_client(), MODEL, unit, industry_context)

            # Persist
            from pathlib import Path as _Path
            unit_data = unit.model_dump()
            unit_data["knowledge_questions"] = questions
            pkg_dir = _Path("units") / unit.training_package.lower()
            pkg_dir.mkdir(parents=True, exist_ok=True)
            (_Path(pkg_dir) / f"{unit_code}.json").write_text(
                json.dumps(unit_data, indent=2))

            # Firestore
            gcp = os.getenv("GOOGLE_CLOUD_PROJECT")
            if gcp:
                try:
                    from google.cloud import firestore as _fs
                    _db = _fs.Client(project=gcp)
                    _db.collection("rpl_unit_registry").document(unit_code).update(
                        {"knowledge_questions": questions})
                except Exception as fe:
                    logger.warning(f"Firestore enrich {unit_code}: {fe}")

            # Reload into registry
            from .unit_registry import UnitOfCompetency as _UoC
            registry.add(_UoC(**unit_data))
            logger.info(f"Enriched {unit_code} — {len(questions)} questions generated")

        except Exception as e:
            logger.warning(f"Enrich failed for {unit_code}: {e}")
        finally:
            _enrich_queue.task_done()
        # Small delay to avoid hammering the AI API
        await asyncio.sleep(2)
# CORS — the app authenticates with Bearer tokens (not cookies). When no
# explicit allow-list is configured we use the "*" wildcard WITHOUT credentials
# ("*" + allow_credentials=True is an invalid combination that browsers reject).
# Set ALLOWED_ORIGINS to a comma-separated list to lock this down.
_allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _allowed_origins:
    app.add_middleware(CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
else:
    app.add_middleware(CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        # Note: HSTS is intentionally omitted — Cloud Run manages HTTPS/TLS.
        # Adding HSTS here causes Chrome to permanently refuse connections if
        # the domain or certificate changes, with no user bypass available.
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"]        = "SAMEORIGIN"
        response.headers["Referrer-Policy"]        = "strict-origin-when-cross-origin"
        return response

app.add_middleware(SecurityHeadersMiddleware)

_ac = None
def get_client():
    global _ac
    if _ac is None:
        # Resolve project ID from the environment (Cloud Run injects
        # GOOGLE_CLOUD_PROJECT automatically). Fail loudly if it is missing
        # rather than leaking a hard-coded project ID in source.
        project_id = (os.getenv("GOOGLE_CLOUD_PROJECT") or
                      os.getenv("ANTHROPIC_VERTEX_PROJECT_ID") or
                      os.getenv("GCLOUD_PROJECT"))
        if not project_id:
            raise RuntimeError(
                "No GCP project configured — set GOOGLE_CLOUD_PROJECT "
                "(or ANTHROPIC_VERTEX_PROJECT_ID) before using AI features.")
        region = os.getenv("VERTEX_REGION", "global")
        logger.info(f"Initialising Vertex client: project={project_id} region={region}")
        _ac = anthropic.AnthropicVertex(
            project_id=project_id,
            region=region)
    return _ac

MODEL = "claude-sonnet-4-6"

# ── Auth model ────────────────────────────────────────────────────────────────
# v4 used a shared TRAINER_PIN. v5 uses per-user JWT login.
# verify_trainer() is preserved only for one legacy endpoint shape — every
# trainer route now declares `user: dict = Depends(current_user)` directly.
def verify_trainer(*_args, **_kwargs):  # noqa: D401 — legacy shim
    """Deprecated. Replaced by Depends(current_user)/Depends(require_admin)."""
    return None


# ── Tenant access helpers ─────────────────────────────────────────────────────

def _check_record_tenant(record: dict, user: dict, *,
                          allow_trainer_cross: bool = False):
    """
    Raise 404 if the record is not visible to this user.
    Rules:
      - The record must belong to the user's org.
      - Trainers can only see their own assessments unless allow_trainer_cross.
      - Admins can see everything in their org.
    Uses 404 (not 403) so we don't reveal record existence across tenants.
    """
    if not record:
        raise HTTPException(404, "Record not found.")
    # Super-admin (platform owner) can see everything.
    if user.get("role") == "superadmin":
        return record
    rec_org = record.get("org_id", "")
    # Fail closed: a record must carry an org_id matching the caller's org.
    # Legacy records with no org_id are NOT visible to any org user (only the
    # super-admin above) — otherwise they would leak across every tenant.
    if not rec_org or rec_org != user.get("org_id"):
        raise HTTPException(404, "Record not found.")
    if user.get("role") == "trainer" and not allow_trainer_cross:
        rec_uid = record.get("trainer_user_id", "")
        if rec_uid and rec_uid != user.get("id"):
            raise HTTPException(404, "Record not found.")
    return record


async def _verify_student_token(token: str, assessment_id: Optional[str] = None) -> dict:
    """
    Authenticate a student request by its invite token.

    The invite token is the candidate's credential. When an assessment_id is
    supplied, the token MUST resolve to that same assessment — this prevents a
    holder of one valid token from reading or mutating another candidate's
    record (IDOR). Returns the assessment record on success.
    """
    if not token:
        raise HTTPException(401, "Missing invite token.")
    record = await get_by_token(token)
    if not record:
        raise HTTPException(404, "Invalid or expired invite link.")
    if assessment_id is not None and record.get("assessment_id") != assessment_id:
        # Same 404 as an unknown token — don't confirm the id exists.
        raise HTTPException(404, "Invalid or expired invite link.")
    return record


def _scope_args_for(user: dict) -> dict:
    """Return kwargs for list_assessments(...) to scope by tenant + role."""
    if user.get("role") == "superadmin":
        return {}  # platform owner sees everything
    args = {"org_id": user.get("org_id")}
    if user.get("role") == "trainer":
        args["trainer_user_id"] = user.get("id")
    return args


# ── Models ─────────────────────────────────────────────────────────────────────
class CreateAssessmentRequest(BaseModel):
    trainer_name: str
    trainer_email: str
    unit_codes: list[str]           # one or more units
    candidate: dict                 # name, email, employer, role, etc.
    notes: str = ""                 # trainer notes for the student

class ProgressSaveRequest(BaseModel):
    assessment_id: str
    progress: dict                  # full progress object
    token: str = ""                 # invite token — authenticates the candidate

class KnowledgeAnalysisRequest(BaseModel):
    assessment_id: str
    unit_code: str
    question: str
    answer: str
    pc_refs: list
    element_ref: str
    q_num: int = 1
    token: str = ""                 # invite token — authenticates the candidate

class MappingRequest(BaseModel):
    assessment_id: str
    unit_code: str
    candidate: dict
    evidence_summary: str
    knowledge_responses: dict = {}
    checklist_results: dict = {}
    use_orchestrator: bool = True   # set False to use legacy single-agent mapping
    industry_context: str = ""
    uploads: dict = {}
    candidate_notes: dict = {}

class TTSRequest(BaseModel):
    text: str
    voice: str = "Puck"
    speaking_rate: float = 0.93
    token: str = ""                 # invite token — gates access to paid TTS

class BulkCreateRequest(BaseModel):
    trainer_name: str
    trainer_email: str
    candidates: list          # list of {name, email, employer, role, duration}
    unit_codes: list[str]
    notes: str = ""

class TemplateCreate(BaseModel):
    name: str
    description: str
    unit_codes: list[str]
    trainer_email: str

class CrossUnitRequest(BaseModel):
    assessment_id: str
    unit_codes: list[str]
    candidate: dict
    evidence_summary: str
    knowledge_responses: dict = {}

class ThirdPartyReportRequest(BaseModel):
    assessment_id: str
    unit_code: str
    candidate: dict
    mapping: dict


class AssessorDecision(BaseModel):
    assessment_id: str
    pc_id: str
    assessor_verdict: str
    assessor_notes: str

class DeterminationRequest(BaseModel):
    assessment_id: str
    unit_code: str
    industry_context: str = ""
    pc_determinations: list   # [{pc_id, assessor_judgement, assessor_notes, override_reason}]
    overall_determination: str  # "RPL Granted" | "RPL Partially Granted" | "RPL Not Granted"
    assessor_rationale: str
    reasonable_adjustments: str = ""
    assessor_name: str
    assessor_id: str = ""

class PreScreenRequest(BaseModel):
    unit_codes: list[str]
    candidate: dict           # name, employer, role, duration, qualifications
    resume_text: str = ""
    industry_context: str = ""

class EvidenceSummaryRequest(BaseModel):
    assessment_id: str
    unit_code: str
    industry_context: str = ""

class IndustryContextRequest(BaseModel):
    assessment_id: str
    industry_context: str
    industry_sector: str = ""


# ── Health ─────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Restore uploaded units from Firestore and start background enrichment worker."""
    global _enrich_queue

    # Fail closed: in a managed/production environment (Cloud Run sets K_SERVICE)
    # an unset AUTH_SECRET means every instance signs JWTs with its own ephemeral
    # key, silently breaking auth and degrading security. Refuse to start instead.
    if not os.getenv("AUTH_SECRET") and os.getenv("K_SERVICE"):
        raise RuntimeError(
            "AUTH_SECRET must be set in production (Cloud Run). "
            "Generate one with: python -c \"import secrets;print(secrets.token_urlsafe(48))\"")

    from .unit_registry import sync_registry_from_firestore
    from .database import _firestore, _token_index
    await sync_registry_from_firestore()
    logger.info(f"Startup complete — {registry.count} units in registry")

    # Rebuild token index from Firestore so existing assessments survive redeploy
    db = _firestore()
    if db:
        try:
            rebuilt = 0
            async for doc in db.collection("rpl_token_index").stream():
                data = doc.to_dict()
                if data.get("assessment_id"):
                    _token_index[doc.id] = data["assessment_id"]
                    rebuilt += 1
            logger.info(f"Token index rebuilt: {rebuilt} tokens loaded")
        except Exception as e:
            logger.warning(f"Token index rebuild failed: {e} — falling back to query scan")
            # Fallback: scan all assessments and build index
            try:
                async for doc in db.collection("rpl_assessments").stream():
                    data = doc.to_dict()
                    tok = data.get("invite_token")
                    aid = data.get("assessment_id")
                    if tok and aid:
                        _token_index[tok] = aid
                        # Write to token_index collection for future startups
                        try:
                            await db.collection("rpl_token_index").document(tok).set({
                                "assessment_id": aid,
                                "created_at": data.get("created_at","")
                            })
                        except Exception:
                            pass
                logger.info(f"Token index rebuilt from scan: {len(_token_index)} tokens")
            except Exception as e2:
                logger.error(f"Token index scan also failed: {e2}")

    # Start background question enrichment worker
    _enrich_queue = asyncio.Queue(maxsize=500)
    asyncio.create_task(_enrich_worker())
    logger.info("Background question enrichment worker started")


@app.get("/health")
async def health():
    from .database import _firestore, _token_index as _tidx
    fs_status = "unconfigured"
    fs_count  = 0
    db = _firestore()
    if db:
        try:
            test_ref = db.collection("rpl_health").document("ping")
            await test_ref.set({"ts": datetime.now(timezone.utc).isoformat()})
            doc = await test_ref.get()
            fs_status = "ok" if doc.exists else "write_failed"
            async for _ in db.collection("rpl_assessments").limit(200).stream():
                fs_count += 1
        except Exception as e:
            fs_status = f"error:{str(e)[:100]}"
    return {
        "status":                  "ok",
        "version":                 "5.0.0",
        "units_loaded":            registry.count,
        "firestore":               fs_status,
        "assessments_in_firestore": fs_count,
        "tokens_in_memory":        len(_tidx),
        "auth":                    await auth_stats(),
        "timestamp":               datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION & ACCOUNT
# Every trainer/admin route below requires a Bearer JWT from /api/auth/login.
# Students still authenticate by unique invite token (unchanged).
# ══════════════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    email:    str
    password: str

class BootstrapRequest(BaseModel):
    bootstrap_key:  str
    org_name:       str
    rto_code:       str = ""
    admin_email:    str
    admin_password: str
    admin_name:     str

class UserCreateRequest(BaseModel):
    email:    str
    password: str
    name:     str
    role:     str = Field(..., pattern="^(admin|trainer)$")

class UserPatchRequest(BaseModel):
    name:     Optional[str] = None
    role:     Optional[str] = None
    active:   Optional[bool] = None
    password: Optional[str] = None

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password:     str


@app.post("/api/auth/login")
async def auth_login(req: LoginRequest):
    """Email + password → JWT bearer token (12 hours)."""
    user = await authenticate(req.email, req.password)
    if not user:
        # Same error for unknown email and wrong password — don't enumerate.
        raise HTTPException(401, "Invalid email or password.")
    token = issue_token(user)
    org   = await auth_get_org(user["org_id"])
    return {
        "token":    token,
        "expires_hours": int(os.getenv("AUTH_JWT_HOURS", "12")),
        "user":     user,
        "org":      org,
    }


@app.get("/api/auth/me")
async def auth_me(user: dict = Depends(current_user)):
    """Return the current user + their organisation."""
    org = await auth_get_org(user["org_id"])
    return {"user": user, "org": org}


@app.post("/api/auth/change-password")
async def auth_change_password(req: ChangePasswordRequest,
                                user: dict = Depends(current_user)):
    """Change own password — must supply current password."""
    # Re-authenticate to confirm current password
    confirm = await authenticate(user["email"], req.current_password)
    if not confirm:
        raise HTTPException(401, "Current password is incorrect.")
    try:
        await auth_update_user(user["id"], password=req.new_password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"changed": True}


@app.post("/api/auth/bootstrap")
async def auth_bootstrap(req: BootstrapRequest):
    """
    One-time setup: create the first organisation and its first admin user.
    Requires BOOTSTRAP_KEY env var to match. Disable BOOTSTRAP_KEY once done.
    Safe to call again — it just creates another org (with its own admin).
    """
    return await bootstrap_first_org(
        name=req.org_name, rto_code=req.rto_code,
        admin_email=req.admin_email, admin_password=req.admin_password,
        admin_name=req.admin_name,
        provided_key=req.bootstrap_key)


# ── Org admin: users (Administration & Compliance only) ───────────────────────

@app.get("/api/org/users")
async def org_list_users(user: dict = Depends(require_admin)):
    """List all users in the admin's organisation."""
    users = await auth_list_users(user["org_id"])
    return {"users": users, "count": len(users)}


@app.post("/api/org/users")
async def org_create_user(req: UserCreateRequest,
                           user: dict = Depends(require_admin)):
    """Create a new user (admin or trainer) within the admin's organisation."""
    try:
        new = await auth_create_user(
            org_id=user["org_id"], email=req.email, password=req.password,
            name=req.name, role=req.role)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return new


@app.patch("/api/org/users/{user_id}")
async def org_update_user(user_id: str, req: UserPatchRequest,
                           user: dict = Depends(require_admin)):
    """Update a user's name, role, active status, or password.
    Admins can only modify users in their own organisation."""
    target = await _auth.get_user(user_id)
    if not target or target.get("org_id") != user["org_id"]:
        raise HTTPException(404, "User not found.")
    try:
        updated = await auth_update_user(
            user_id,
            name=req.name, role=req.role,
            active=req.active, password=req.password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return updated


@app.delete("/api/org/users/{user_id}")
async def org_delete_user(user_id: str,
                           user: dict = Depends(require_admin)):
    """Remove a user. Admins cannot delete themselves."""
    if user_id == user["id"]:
        raise HTTPException(400, "You cannot delete your own account.")
    target = await _auth.get_user(user_id)
    if not target or target.get("org_id") != user["org_id"]:
        raise HTTPException(404, "User not found.")
    await auth_delete_user(user_id)
    return {"deleted": True}


@app.get("/api/org")
async def org_get(user: dict = Depends(current_user)):
    """Get the current organisation (any role can read)."""
    org = await auth_get_org(user["org_id"])
    if not org:
        raise HTTPException(404, "Organisation not found.")
    return org


@app.patch("/api/org/settings")
async def org_update_settings(settings: dict,
                               user: dict = Depends(require_admin)):
    """Update org settings (e.g. retention years, currency years)."""
    org = await auth_update_org(user["org_id"], settings)
    if not org:
        raise HTTPException(404, "Organisation not found.")
    return org


# ══════════════════════════════════════════════════════════════════════════════
# SUPER-ADMIN (platform owner) — manage all organisations
# Only users with role == "superadmin" can reach these.
# ══════════════════════════════════════════════════════════════════════════════

class OrgCreateRequest(BaseModel):
    org_name:       str
    rto_code:       str = ""
    admin_email:    str
    admin_password: str
    admin_name:     str

class OrgActiveRequest(BaseModel):
    active: bool


@app.get("/api/superadmin/orgs")
async def sa_list_orgs(user: dict = Depends(require_superadmin)):
    """List every organisation on the platform, with user counts."""
    orgs = await list_orgs_with_counts()
    return {"orgs": orgs, "count": len(orgs)}


@app.post("/api/superadmin/orgs")
async def sa_create_org(req: OrgCreateRequest,
                         user: dict = Depends(require_superadmin)):
    """Create a new RTO (organisation) plus its first admin user."""
    result = await create_org_with_admin(
        org_name=req.org_name, rto_code=req.rto_code,
        admin_email=req.admin_email, admin_password=req.admin_password,
        admin_name=req.admin_name)
    logger.info(f"[superadmin {user['email']}] created org "
                f"{result['org']['name']} ({result['org']['id'][:8]})")
    return result


@app.get("/api/superadmin/orgs/{org_id}/users")
async def sa_list_org_users(org_id: str,
                             user: dict = Depends(require_superadmin)):
    """List all users in any organisation."""
    org = await auth_get_org(org_id)
    if not org:
        raise HTTPException(404, "Organisation not found.")
    users = await auth_list_users(org_id)
    return {"org": org, "users": users, "count": len(users)}


@app.patch("/api/superadmin/orgs/{org_id}/active")
async def sa_set_org_active(org_id: str, req: OrgActiveRequest,
                             user: dict = Depends(require_superadmin)):
    """Enable or suspend an entire organisation."""
    org = await set_org_active(org_id, req.active)
    if not org:
        raise HTTPException(404, "Organisation not found.")
    return org


@app.post("/api/superadmin/orgs/{org_id}/admins")
async def sa_add_org_admin(org_id: str, req: UserCreateRequest,
                            user: dict = Depends(require_superadmin)):
    """Add an admin (or trainer) to any organisation."""
    org = await auth_get_org(org_id)
    if not org:
        raise HTTPException(404, "Organisation not found.")
    try:
        new = await auth_create_user(
            org_id=org_id, email=req.email, password=req.password,
            name=req.name, role=req.role)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return new


@app.get("/api/superadmin/stats")
async def sa_stats(user: dict = Depends(require_superadmin)):
    """Platform-wide totals for the super-admin dashboard."""
    orgs = await list_orgs_with_counts()
    total_users    = sum(o.get("user_count", 0) for o in orgs)
    total_trainers = sum(o.get("trainer_count", 0) for o in orgs)
    # Count assessments per org across the platform
    all_assessments = await list_assessments(limit=10000)
    by_org = {}
    for a in all_assessments:
        by_org[a.get("org_id", "")] = by_org.get(a.get("org_id", ""), 0) + 1
    return {
        "total_orgs":       len(orgs),
        "active_orgs":      sum(1 for o in orgs if o.get("active", True)),
        "total_users":      total_users,
        "total_trainers":   total_trainers,
        "total_assessments": len(all_assessments),
        "assessments_by_org": by_org,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINER ENDPOINTS (require Bearer JWT — see /api/auth/login)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/trainer/assessments/create")
async def trainer_create_assessment(
    req: CreateAssessmentRequest,
    user: dict = Depends(current_user)
):
    """
    Trainer creates a new RPL assessment for a candidate.
    Returns assessment_id and the unique invite URL to send to the student.
    The trainer identity is taken from the logged-in user, not the request body.
    """
    # Validate all unit codes exist
    for code in req.unit_codes:
        if not registry.get(code):
            raise HTTPException(404, f"Unit {code} not found. Import it first.")

    # Identity is from the JWT, not from req fields — prevents spoofing.
    trainer_name  = user.get("name") or req.trainer_name
    trainer_email = user.get("email") or req.trainer_email
    trainer_id    = f"trainer_{trainer_email.replace('@','_').replace('.','_')}"

    assessment_id, invite_token = await create_assessment(trainer_id, {
        "org_id":          user["org_id"],
        "trainer_user_id": user["id"],
        "trainer_name":    trainer_name,
        "trainer_email":   trainer_email,
        "unit_codes":      req.unit_codes,
        "candidate":       req.candidate,
        "notes":           req.notes,
    })

    base_url = os.getenv("BASE_URL", "")
    invite_url = f"{base_url}/rpl/{invite_token}"

    logger.info(f"[org={user['org_id']}] Assessment {assessment_id} created "
                f"for {req.candidate.get('name')} — units: {req.unit_codes}")

    # Auto-send invite email to candidate
    await _notify_candidate_invite({
        "candidate":     req.candidate,
        "trainer_name":  trainer_name,
        "trainer_email": trainer_email,
        "unit_codes":    req.unit_codes,
        "invite_url":    invite_url,
        "assessment_id": assessment_id,
        "notes":         req.notes,
    })

    # Get the record back to check if Firestore write succeeded
    created = await get_assessment(assessment_id)
    fs_ok = created.get("_firestore_ok", False) if created else False
    email_sent = bool(os.getenv("SENDGRID_API_KEY"))
    return {
        "assessment_id":   assessment_id,
        "invite_token":    invite_token,
        "invite_url":      invite_url,
        "candidate":       req.candidate,
        "unit_codes":      req.unit_codes,
        "status":          "INVITED",
        "email_sent":      email_sent,
        "firestore_saved": fs_ok,
        "message":         (f"Invite emailed to {req.candidate.get('email')}"
                            if email_sent else f"Copy this URL to send to the candidate: {invite_url}"),
        "warning":         ("" if fs_ok else
                            "⚠ Assessment saved to memory only — Firestore write failed. "
                            "This assessment will be lost if the server restarts. "
                            "Check /health for Firestore status."),
    }


@app.get("/api/trainer/assessments")
async def trainer_list_assessments(
    status: Optional[str] = None,
    user: dict = Depends(current_user)
):
    """List assessments for this user.
    - Trainers see only their own.
    - Admins (Administration & Compliance) see all assessments in the org.
    """
    results = await list_assessments(status=status, **_scope_args_for(user))
    return {"assessments": results, "count": len(results),
            "scope": "org" if user.get("role") == "admin" else "self"}


@app.get("/api/trainer/assessments/{assessment_id}")
async def trainer_get_assessment(
    assessment_id: str,
    user: dict = Depends(current_user)
):
    """Get full assessment record including student progress (tenant-scoped)."""
    data = await get_assessment(assessment_id)
    _check_record_tenant(data, user)
    return data


@app.post("/api/trainer/assessments/{assessment_id}/complete")
async def trainer_complete_assessment(
    assessment_id: str,
    decision: AssessorDecision,
    user: dict = Depends(current_user)
):
    """Trainer marks assessment as complete with final determination."""
    rec = await get_assessment(assessment_id)
    _check_record_tenant(rec, user)
    await save_assessment(assessment_id, f"assessor_{decision.pc_id}", {
        "pc_id": decision.pc_id,
        "assessor_verdict": decision.assessor_verdict,
        "assessor_notes": decision.assessor_notes,
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "decided_by": {"user_id": user["id"], "name": user.get("name"),
                       "email": user.get("email"), "role": user.get("role")},
        "source": "HUMAN_ASSESSOR", "hitl_compliant": True
    })
    await complete_assessment(assessment_id)
    return {"saved": True, "status": "COMPLETE"}


@app.post("/api/trainer/assessments/{assessment_id}/resend-invite")
async def resend_invite(
    assessment_id: str,
    user: dict = Depends(current_user)
):
    """Get the invite URL again to resend."""
    data = await get_assessment(assessment_id)
    _check_record_tenant(data, user)
    base_url = os.getenv("BASE_URL", "")
    token = data.get("invite_token")
    return {"invite_url": f"{base_url}/rpl/{token}", "candidate": data.get("candidate")}


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT ENDPOINTS (authenticated by invite token)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/admin/token-debug/{token}")
async def token_debug(token: str, user: dict = Depends(require_admin)):
    """Diagnose why a token lookup might be failing."""
    from .database import _firestore, _token_index as _tidx
    result = {
        "token_prefix":     token[:8],
        "in_memory_index":  token in _tidx,
        "memory_index_size": len(_tidx),
        "firestore_available": False,
        "token_index_hit":   False,
        "query_hit":         False,
        "scan_hit":          False,
        "assessment_id":     None,
    }
    db = _firestore()
    result["firestore_available"] = db is not None
    if db:
        # Check token_index collection
        try:
            idx = await db.collection("rpl_token_index").document(token).get()
            result["token_index_hit"] = idx.exists
            if idx.exists:
                result["assessment_id"] = idx.to_dict().get("assessment_id")
        except Exception as e:
            result["token_index_error"] = str(e)

        # Try query
        try:
            count = 0
            async for doc in db.collection("rpl_assessments").where(
                    "invite_token", "==", token).stream():
                count += 1
                result["query_hit"] = True
                result["assessment_id"] = doc.to_dict().get("assessment_id")
            result["query_count"] = count
        except Exception as e:
            result["query_error"] = str(e)

        # Count total assessments
        try:
            total = 0
            async for _ in db.collection("rpl_assessments").stream():
                total += 1
            result["total_assessments_in_firestore"] = total
        except Exception as e:
            result["count_error"] = str(e)
    return result


@app.get("/api/student/join/{token}")
async def student_join(token: str):
    """
    Student opens invite link. Returns assessment data with progress.
    Uses layered lookup: token_index → query → full scan.
    """
    from .database import _firestore, _token_index as _tidx

    data = await get_by_token(token)

    # Hard fallback: direct Firestore full scan right here in the endpoint
    # Catches cases where startup didn't finish indexing or Firestore was slow
    if not data:
        logger.warning(f"get_by_token missed for {token[:8]}... — running direct scan")
        db = _firestore()
        if db:
            try:
                async for doc in db.collection("rpl_assessments").stream():
                    rec = doc.to_dict()
                    if rec and rec.get("invite_token") == token:
                        data = rec
                        # Backfill index so next lookup is instant
                        _tidx[token] = rec["assessment_id"]
                        try:
                            await db.collection("rpl_token_index").document(token).set({
                                "assessment_id": rec["assessment_id"],
                                "created_at":    rec.get("created_at", ""),
                            })
                        except Exception:
                            pass
                        logger.info(f"Token {token[:8]}... found via direct scan — index backfilled")
                        break
            except Exception as e:
                logger.error(f"Direct scan failed: {e}")

    if not data:
        raise HTTPException(404, "Invalid or expired invite link.")

    # Update status to IN_PROGRESS on first open
    if data.get("status") == "INVITED":
        await save_assessment(data["assessment_id"], "status_update",
            {"status": "IN_PROGRESS", "first_opened": datetime.now(timezone.utc).isoformat()})

    # Return full assessment including saved progress
    units = []
    for code in data.get("unit_codes", []):
        unit = registry.get(code)
        if unit:
            units.append(unit.model_dump())

    return {
        "assessment_id": data["assessment_id"],
        "token":         token,
        "candidate":     data.get("candidate", {}),
        "trainer_name":  data.get("trainer_name", ""),
        "notes":         data.get("notes", ""),
        "unit_codes":    data.get("unit_codes", []),
        "units":         units,
        "status":        data.get("status", "IN_PROGRESS"),
        "progress":      data.get("progress", {}),
        "created_at":    data.get("created_at", ""),
    }


@app.post("/api/student/progress")
async def student_save_progress(req: ProgressSaveRequest):
    """
    Auto-save student progress after every step.
    Called from the frontend whenever anything changes.
    The invite token authenticates the candidate and must own this assessment.
    """
    await _verify_student_token(req.token, req.assessment_id)

    await save_progress(req.assessment_id, req.progress)
    return {"saved": True, "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/api/student/submit/{assessment_id}")
async def student_submit(assessment_id: str, token: str = ""):
    """Student submits their completed RPL — notifies trainer.
    The invite token authenticates the candidate and must own this assessment."""
    data = await _verify_student_token(token, assessment_id)

    await submit_assessment(assessment_id)

    # Send email notification to trainer (if SendGrid configured)
    await _notify_trainer_submission(data)

    return {"submitted": True, "message": "Your RPL has been submitted. Your trainer will review it and be in touch."}


async def _notify_trainer_submission(assessment_data: dict):
    """Send email to trainer when student submits. Requires SENDGRID_API_KEY."""
    api_key = os.getenv("SENDGRID_API_KEY")
    if not api_key:
        logger.info("SendGrid not configured — skipping email notification")
        return
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        candidate = assessment_data.get("candidate", {})
        trainer_email = assessment_data.get("trainer_email", "")
        if not trainer_email:
            return
        msg = Mail(
            from_email=os.getenv("SENDGRID_FROM", "rpl@abctraining.com.au"),
            to_emails=trainer_email,
            subject=f"RPL submitted — {candidate.get('name', 'Candidate')}",
            html_content=f"""
            <p>Hi {assessment_data.get('trainer_name', 'Trainer')},</p>
            <p><strong>{candidate.get('name')}</strong> has submitted their RPL assessment for review.</p>
            <p>Units: {', '.join(assessment_data.get('unit_codes', []))}</p>
            <p>Please log in to the trainer dashboard to review and provide your determination.</p>
            <p>ABC Training | RTO #5800</p>""")
        sg = SendGridAPIClient(api_key)
        sg.send(msg)
        logger.info(f"Submission notification sent to {trainer_email}")
    except Exception as e:
        logger.warning(f"Email notification failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# UNIT REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/units")
async def list_units():
    return {"units": registry.list_all(), "count": registry.count}

@app.get("/api/units/{unit_code}")
async def get_unit(unit_code: str):
    unit = registry.get(unit_code)
    if not unit: raise HTTPException(404, f"Unit {unit_code} not found")
    return unit.model_dump()

@app.post("/api/units/import/{unit_code}")
async def import_unit(unit_code: str, user: dict = Depends(require_admin)):
    existing = registry.get(unit_code)
    if existing:
        return {"message": f"{unit_code} already in registry", "unit": existing.model_dump()}
    try:
        unit = await import_from_tgau(unit_code)
        registry.add(unit)
        return {"message": f"Imported {unit_code}", "unit": unit.model_dump()}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Import failed: {e}")

@app.get("/api/units/debug/{unit_code}")
async def debug_tga_import(unit_code: str, user: dict = Depends(require_admin)):
    """
    Debug endpoint — tests TGA SOAP and scraper without saving.
    Shows exactly what each method returns so you can diagnose failures.
    """
    import xml.etree.ElementTree as ET

    code = unit_code.upper().strip()
    # TGA web-service credentials must come from the environment — never hard-code
    # them in source. If unset, the SOAP probe simply runs unauthenticated.
    tga_user = os.getenv("TGA_USER", "")
    tga_pass = os.getenv("TGA_PASS", "")
    result    = {"code": code, "soap": {}, "scraper": {}}

    # ── Test SOAP ──────────────────────────────────────────────────────────────
    soap_url = "https://ws.training.gov.au/Deewr.Tga.Webservices/TrainingComponentServiceV12.svc/Training"
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"
            xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">
  <s:Header>
    <wsse:Security>
      <wsse:UsernameToken>
        <wsse:Username>{tga_user}</wsse:Username>
        <wsse:Password>{tga_pass}</wsse:Password>
      </wsse:UsernameToken>
    </wsse:Security>
  </s:Header>
  <s:Body>
    <GetDetails xmlns="http://training.gov.au/">
      <request>
        <Code>{code}</Code>
        <ShowReleases>true</ShowReleases>
        <ShowUnitGrid>true</ShowUnitGrid>
        <ShowComponents>true</ShowComponents>
      </request>
    </GetDetails>
  </s:Body>
</s:Envelope>"""

    try:
        async with httpx.AsyncClient(timeout=20) as http:
            r = await http.post(soap_url, content=soap_body.encode(),
                headers={"Content-Type": "text/xml; charset=utf-8",
                         "SOAPAction": "http://training.gov.au/ITrainingComponentService/GetDetails"})
            result["soap"]["status"] = r.status_code
            result["soap"]["body_length"] = len(r.text)
            result["soap"]["body_preview"] = r.text[:800]
            result["soap"]["success"] = r.is_success
    except Exception as e:
        result["soap"]["error"] = str(e)

    # ── Test scraper URLs ──────────────────────────────────────────────────────
    urls = [
        f"https://training.gov.au/training/details/{code}/unitdetails",
        f"https://training.gov.au/Training/Details/{code}",
    ]
    result["scraper"]["urls_tested"] = []
    for url in urls:
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}) as http:
                r = await http.get(url)
                result["scraper"]["urls_tested"].append({
                    "url": url, "status": r.status_code,
                    "body_length": len(r.text),
                    "body_preview": r.text[:300],
                    "final_url": str(r.url),
                })
        except Exception as e:
            result["scraper"]["urls_tested"].append({"url": url, "error": str(e)})

    return result


@app.post("/api/units/create")
async def create_unit(data: dict, user: dict = Depends(require_admin)):
    try:
        unit = UnitOfCompetency(**data); registry.add(unit)
        return {"message": f"Created {unit.code}", "unit": unit.model_dump()}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/admin/upload-training-package")
async def upload_training_package(
    file: UploadFile = File(...),
    user: dict = Depends(require_admin)
):
    """
    Upload a training.gov.au Excel export file to load units into the registry.
    Accepts the standard TGA export format with columns:
    UoC Code, UoC Title, Usage Recommendation, IsConfidential,
    TP Scheme, TP Code, TP Title, Content Item Name, Contents
    """
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(400, "File must be an Excel file (.xlsx or .xls)")

    try:
        import io, re
        import openpyxl

        content = await file.read()
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True)

        # Find the Export sheet
        sheet_name = "Export" if "Export" in wb.sheetnames else wb.sheetnames[0]
        ws = wb[sheet_name]
        rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r and r[0]]

        def parse_elements_pcs(text):
            if not text: return []
            elements = []
            text = text.replace("\r", "").replace("\t", " ")
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            skip = ["elements describe", "performance criteria describe",
                    "performance needed", "essential outcomes",
                    "bold and italicised", "bold italicised", "range of conditions"]
            el_num = 0; current_el = None; current_pcs = []
            i = 0
            while i < len(lines):
                line = lines[i]
                if any(p in line.lower() for p in skip): i+=1; continue
                if line.lower() in ["elements", "performance criteria"]: i+=1; continue

                # PC: "1.1 text" on same line
                pm = re.match(r"^(\d+)\.(\d+)[\. ]+(.+)$", line)
                if pm and current_el is not None:
                    _pc_text = pm.group(3).strip()
                    current_pcs.append({"id": f"{pm.group(1)}.{pm.group(2)}",
                                        "text": _pc_text,
                                        "element_id": f"E{el_num}",
                                        "analysis_prompt": f"What valid, sufficient, authentic and current evidence shows the candidate can {_pc_text.rstrip('.')}?",
                                        "benchmark_statement": f"The candidate demonstrates that they can {_pc_text.rstrip('.')}.",
                                        "evidence_record": {"evidence_items": [], "assessor_comments": "", "judgement": None, "gap_notes": ""}})
                    i+=1; continue

                # PC: "1.1" alone, text on next line (MSL/split format)
                pm2 = re.match(r"^(\d+)\.(\d+)$", line)
                if pm2 and current_el is not None and i+1 < len(lines):
                    nxt = lines[i+1].strip()
                    if not re.match(r"^\d+[\d\.]*$", nxt):
                        current_pcs.append({"id": f"{pm2.group(1)}.{pm2.group(2)}",
                                            "text": nxt, "element_id": f"E{el_num}",
                                            "analysis_prompt": f"What valid, sufficient, authentic and current evidence shows the candidate can {nxt.rstrip('.')}?",
                                            "benchmark_statement": f"The candidate demonstrates that they can {nxt.rstrip('.')}.",
                                            "evidence_record": {"evidence_items": [], "assessor_comments": "", "judgement": None, "gap_notes": ""}})
                        i+=2; continue

                # Element: "1. Title" or "1 Title" same line
                em = re.match(r"^(\d+)\.?\s+(.+)$", line)
                if em and int(em.group(1)) == el_num + 1:
                    if current_el:
                        elements.append({"id": f"E{el_num}", "title": current_el, "pcs": current_pcs,
                                       "analysis_focus": f"Analyse evidence against Element {el_num}: {current_el}"})
                    el_num = int(em.group(1)); current_el = em.group(2).strip(); current_pcs = []
                    i+=1; continue

                # Element: bare number alone, title on next line (MSL/split format)
                em2 = re.match(r"^(\d+)$", line)
                if em2 and int(em2.group(1)) == el_num + 1 and i+1 < len(lines):
                    nxt = lines[i+1].strip()
                    if not re.match(r"^\d+[\d\.]*$", nxt):
                        if current_el:
                            elements.append({"id": f"E{el_num}", "title": current_el, "pcs": current_pcs,
                                               "analysis_focus": f"Analyse evidence against Element {el_num}: {current_el}"})
                        el_num = int(em2.group(1)); current_el = nxt; current_pcs = []
                        i+=2; continue

                i+=1

            if current_el:
                elements.append({"id": f"E{el_num}", "title": current_el, "pcs": current_pcs,
                                   "analysis_focus": f"Analyse evidence against Element {el_num}: {current_el}"})
            return elements

        added   = 0
        updated = 0
        skipped = 0
        errors  = []
        newly_processed = []   # track codes for enrichment queue
        packages_seen = set()
        from pathlib import Path

        for row in rows:
            if len(row) < 9: skipped += 1; continue
            code      = (row[0] or "").strip().upper()
            title     = (row[1] or "").strip()
            status    = (row[2] or "").strip()
            pkg_code  = (row[5] or "").strip().upper()
            pkg_title = (row[6] or "").strip()
            contents  = row[8] or ""

            if not code or not title: skipped += 1; continue
            if status.lower() not in ["current", ""]: skipped += 1; continue

            packages_seen.add(f"{pkg_code}: {pkg_title[:50]}")
            is_existing = registry.get(code) is not None

            elements = parse_elements_pcs(contents)
            if not elements:
                elements = [{"id": "E1", "title": f"Apply {title}", "pcs": [
                    {"id": "1.1", "text": "Perform duties consistent with this unit", "element_id": "E1"}
                ]}]

            knowledge_qs = []
            q_num = 1
            for el in elements[:8]:
                for pc in el["pcs"][:3]:
                    ap = pc.get("analysis_prompt", f"What valid evidence shows the candidate can {pc['text'].rstrip('.')}?")
                    bs = pc.get("benchmark_statement", f"The candidate demonstrates that they can {pc['text'].rstrip('.')}.")
                    pt = pc["text"].rstrip(".")
                    pt_lower = pt.lower()
                    # Knowledge checks — NOT competency conversation starters
                    if any(w in pt_lower for w in ["check","confirm","verify","inspect","validate"]):
                        q_text = f"What checks would you perform to {pt_lower}?"
                    elif any(w in pt_lower for w in ["identify","assess","analyse","determine","classify"]):
                        q_text = f"What factors would you consider when you need to {pt_lower}?"
                    elif any(w in pt_lower for w in ["record","document","report","log","complete"]):
                        q_text = f"What information must be recorded when you {pt_lower}, and why?"
                    elif any(w in pt_lower for w in ["hazard","risk","safety","ppe","protection","control"]):
                        q_text = f"What are the key hazards and required controls associated with {pt_lower}?"
                    elif any(w in pt_lower for w in ["select","choose","source","obtain"]):
                        q_text = f"What criteria determine how to {pt_lower}?"
                    elif any(w in pt_lower for w in ["prepare","set up","establish","configure","calibrat"]):
                        q_text = f"What steps are required to {pt_lower}, and what would indicate it is ready to proceed?"
                    elif any(w in pt_lower for w in ["monitor","maintain","ensure","sustain"]):
                        q_text = f"What would indicate that {pt_lower} is not being achieved, and what action should be taken?"
                    elif any(w in pt_lower for w in ["communicate","report","notify","advise","inform"]):
                        q_text = f"What information must be communicated when {pt_lower}, and to whom?"
                    else:
                        q_text = f"What technical knowledge is required to {pt_lower}? Explain the key requirements, procedures, or standards involved."
                    knowledge_qs.append({
                        "num": q_num,
                        "element_ref": f"Element {el['id']}: {el['title']}",
                        "pc_refs": [pc["id"]],
                        "pc_id": pc["id"],
                        "text": q_text,
                        "hint": f"Focus on technical knowledge: {'; '.join(pc.get('knowledge_focus', [bs[:80]]))}",
                        "difficulty_level": "Applied",
                        "question_purpose": f"Tests underpinning knowledge required to {pt_lower}",
                        "why_task_specific": f"Targets the specific technical knowledge needed to perform PC {pc['id']}",
                        "benchmark_statement": bs,
                        "analysis_prompt": ap,
                        "practical_task_interpretation": f"In practice this requires the worker to {pt_lower}",
                        "knowledge_focus": [pt],
                        "workplace_context_examples": [el["title"]],
                        "model_answer_guide": {
                            "expected_knowledge_points": [f"Technical understanding of {pt_lower}"],
                            "acceptable_answer_examples": [],
                            "strong_answer_indicators": ["Specific, technical, workplace-relevant detail with correct terminology"],
                            "weak_answer_indicators": ["Vague, generic, theoretical, or no workplace-specific content"],
                            "common_gaps_or_errors": ["Unable to specify workplace-relevant requirements or conditions"]
                        },
                        "assessor_framework": {
                            "what_to_look_for": [f"Accurate technical knowledge of {pt_lower}"],
                            "minimum_expected_knowledge": [bs],
                            "indicators_of_partial_understanding": ["Knows the general concept but cannot specify technical detail"],
                            "indicators_of_strong_understanding": ["Demonstrates precise, contextual, technically accurate knowledge with workplace application"]
                        }
                    })
                    q_num += 1
                    if q_num > 12: break
                if q_num > 12: break

            evidence_guide = [
                {
                    "title": "Third Party Report — Workplace Supervisor (Required)",
                    "priority": "priority",
                    "pc_refs": [pc["id"] for el in elements[:2] for pc in el["pcs"][:3]],
                    "icon": "📝",
                    "description": "Completed by your direct supervisor — strongest evidence for RPL.",
                    "acceptable_forms": ["Download Third Party Report template",
                                         "Supervisor completes and signs",
                                         "Upload completed signed form"]
                },
                {
                    "title": "Position description",
                    "priority": "recommended",
                    "pc_refs": [el["pcs"][0]["id"] for el in elements if el["pcs"]][:5],
                    "icon": "📋",
                    "description": "Current position description showing duties relevant to this unit.",
                    "acceptable_forms": ["Current employer PD", "Letter from employer confirming duties"]
                }
            ]

            unit_data = {
                "code": code, "title": title,
                "training_package": pkg_code,
                "training_package_name": pkg_title,
                "application": "",
                "competent_person_statement": (
                    f"A competent person in {title} demonstrates consistent "
                    "performance across all elements and performance criteria."),
                "elements": elements,
                "knowledge_requirements": [], "skill_requirements": [],
                "evidence_guide": evidence_guide,
                "knowledge_questions": knowledge_qs,
                "currency_years": 5, "source": "xlsx", "version": "1.0",
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            try:
                unit = UnitOfCompetency(**unit_data)
                registry.add(unit)  # always overwrites in memory

                # Persist to local disk
                pkg_dir = Path("units") / pkg_code.lower()
                pkg_dir.mkdir(parents=True, exist_ok=True)
                (pkg_dir / f"{code}.json").write_text(json.dumps(unit_data, indent=2))

                # Persist to Firestore — survives future redeployments
                gcp_proj = os.getenv("GOOGLE_CLOUD_PROJECT")
                if gcp_proj:
                    try:
                        from google.cloud import firestore as _fs
                        _db = _fs.Client(project=gcp_proj)
                        _db.collection("rpl_unit_registry").document(code).set(unit_data)
                    except Exception as _fe:
                        logger.warning(f"Firestore save failed for {code}: {_fe}")
                if is_existing:
                    updated += 1
                    newly_processed.append(code)
                else:
                    added += 1
                    newly_processed.append(code)
            except Exception as e:
                errors.append(f"{code}: {str(e)[:80]}")
                skipped += 1

        total_pcs = sum(
            sum(len(e.pcs) for e in u.elements)
            for u in registry._units.values()
        )
        parts = []
        if added:   parts.append(f"{added} new")
        if updated: parts.append(f"{updated} updated")
        summary = " · ".join(parts) if parts else "0 units processed"

        # Queue all new/updated units for AI question enrichment
        queued = 0
        if _enrich_queue is not None:
            for code in newly_processed:
                try:
                    _enrich_queue.put_nowait({"unit_code": code, "industry_context": ""})
                    queued += 1
                except asyncio.QueueFull:
                    break

        return {
            "success":               True,
            "filename":              file.filename,
            "imported":              added + updated,
            "added":                 added,
            "updated":               updated,
            "skipped":               skipped,
            "errors":                errors[:10],
            "packages":              sorted(packages_seen),
            "registry_total":        registry.count,
            "total_pcs_in_registry": total_pcs,
            "questions_queued_for_enrichment": queued,
            "message":               f"{summary} from {file.filename} · {queued} units queued for AI question generation"
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")


@app.get("/api/admin/enrichment-status")
async def enrichment_status(user: dict = Depends(require_admin)):
    """How many units are still queued for AI question enrichment."""
    queue_size = _enrich_queue.qsize() if _enrich_queue else 0
    # Count units that already have AI-quality questions
    enriched = sum(
        1 for u in registry._units.values()
        if u.knowledge_questions and
           u.knowledge_questions[0].model_answer_guide.expected_knowledge_points
           and u.knowledge_questions[0].model_answer_guide.expected_knowledge_points[0]
           != f"Technical understanding of {u.knowledge_questions[0].pc_id}"
    )
    return {
        "queue_size":       queue_size,
        "enriched_units":   enriched,
        "total_units":      registry.count,
        "percent_enriched": round(enriched / registry.count * 100, 1) if registry.count else 0,
        "status": "idle" if queue_size == 0 else f"enriching ({queue_size} units remaining)"
    }


@app.get("/api/admin/stats")
async def admin_stats(user: dict = Depends(require_admin)):
    """Registry statistics for the admin panel."""
    from pathlib import Path
    packages = {}
    for unit in registry._units.values():
        pkg = unit.training_package
        if pkg not in packages:
            packages[pkg] = {"code": pkg, "name": unit.training_package_name, "count": 0, "pcs": 0}
        packages[pkg]["count"] += 1
        packages[pkg]["pcs"] += sum(len(e.pcs) for e in unit.elements)

    return {
        "total_units": registry.count,
        "total_packages": len(packages),
        "total_pcs": sum(p["pcs"] for p in packages.values()),
        "packages": sorted(packages.values(), key=lambda x: -x["count"])
    }


@app.delete("/api/units/{unit_code}")
async def delete_unit(unit_code: str, user: dict = Depends(require_admin)):
    if not registry.delete(unit_code): raise HTTPException(404, f"Unit {unit_code} not found")
    return {"message": f"Deleted {unit_code}"}


# ══════════════════════════════════════════════════════════════════════════════
# AI ENDPOINTS (used by both trainer and student portals)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/tts")
async def synthesise_speech(req: TTSRequest):
    import asyncio
    # Gate paid speech synthesis behind a valid invite token so the endpoint
    # can't be used as an open, anonymous proxy to Google TTS (cost abuse).
    await _verify_student_token(req.token)
    # Bound request size — a single utterance, not a document.
    if len(req.text or "") > 5000:
        raise HTTPException(413, "Text too long for synthesis (max 5000 chars).")
    voice_name = f"en-AU-Chirp3-HD-{req.voice}"
    logger.info(f"TTS: {voice_name}, {len(req.text)} chars")
    def _call():
        from google.cloud import texttospeech
        tts = texttospeech.TextToSpeechClient()
        response = tts.synthesize_speech(
            input=texttospeech.SynthesisInput(text=req.text),
            voice=texttospeech.VoiceSelectionParams(language_code="en-AU", name=voice_name),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=req.speaking_rate))
        return base64.b64encode(response.audio_content).decode("utf-8")
    try:
        loop = asyncio.get_event_loop()
        audio_b64 = await loop.run_in_executor(None, _call)
        return {"audioContent": audio_b64, "voice": voice_name}
    except Exception as e:
        msg = str(e)
        if "credentials" in msg.lower() or "default" in msg.lower():
            raise HTTPException(500, "TTS credentials error. Run: gcloud auth application-default login")
        raise HTTPException(500, f"TTS error: {msg}")


@app.post("/api/knowledge/analyse")
async def analyse_knowledge(req: KnowledgeAnalysisRequest):
    # Invite token authenticates the candidate and must own this assessment.
    await _verify_student_token(req.token, req.assessment_id)
    unit = registry.get(req.unit_code)
    if not unit: raise HTTPException(404, f"Unit {req.unit_code} not found")
    if len((req.answer or "").split()) < 5:
        raise HTTPException(400, "Response too short")
    try:
        question_obj = None
        if req.q_num and req.q_num <= len(unit.knowledge_questions):
            question_obj = unit.knowledge_questions[req.q_num - 1]

        # Check if the model_answer_guide has real content — not just a placeholder
        def _has_real_guide(q_obj):
            if not q_obj: return False
            pts = q_obj.model_answer_guide.expected_knowledge_points
            if not pts: return False
            # Placeholder detection: generated guides only have one generic entry
            # starting with "Technical understanding of" — not real knowledge points
            if len(pts) == 1 and pts[0].lower().startswith("technical understanding of"):
                return False
            return True

        # Load candidate for sector-appropriate analysis
        _assessment_rec = await get_assessment(req.assessment_id) if req.assessment_id else {}
        _candidate = (_assessment_rec or {}).get("candidate", {})

        if _has_real_guide(question_obj):
            # Pass candidate so the AI runs sector-aware analysis
            # (previously the candidate was loaded above but not forwarded — bug fix)
            result = await evaluate_knowledge_answer_detailed(
                client=get_client(), model=MODEL, unit=unit,
                question=question_obj.model_dump(),
                answer=req.answer,
                candidate=_candidate)
        else:
            # Fall back to benchmark-driven analysis using PC data directly
            result = await analyse_knowledge_response(
                client=get_client(), model=MODEL, unit=unit,
                question=req.question, answer=req.answer,
                pc_refs=req.pc_refs, element_ref=req.element_ref,
                candidate=_candidate)

            # Queue this unit for AI question enrichment if not already done
            if _enrich_queue is not None and question_obj:
                try:
                    _enrich_queue.put_nowait({
                        "unit_code": req.unit_code,
                        "industry_context": ""
                    })
                except Exception:
                    pass  # Queue full or unavailable — not critical

        # Run AI detection in parallel with knowledge analysis
        assessment = await get_assessment(req.assessment_id) if req.assessment_id else {}
        progress   = (assessment or {}).get("progress", {})
        prior_responses = [
            v for v in (progress.get("knowledge_responses", {})
                        .get(req.unit_code, {}) or {}).values()
            if isinstance(v, str) and v.strip()
        ]
        candidate = (assessment or {}).get("candidate", {})

        q_dict = question_obj.model_dump() if question_obj else {
            "text": req.question, "pc_id": (req.pc_refs or [""])[0], "knowledge_focus": []}
        ai_detection = await detect_ai_usage(
            get_client(), MODEL, unit,
            q_dict, req.answer, prior_responses, candidate)

        result["ai_detection"] = ai_detection

        # Normalise result fields for consistent trainer display
        if "confidence_score_percent" in result and "overall_score_percent" not in result:
            result["overall_score_percent"] = result["confidence_score_percent"]
        if "overall_score_percent" not in result:
            result["overall_score_percent"] = 0
        if "commentary" not in result:
            result["commentary"] = result.get("evaluation_summary") or result.get("summary") or ""
        if "meets_requirement" not in result:
            s = result.get("overall_score_percent", 0)
            result["meets_requirement"] = ("FULLY" if s>=85 else "SUBSTANTIALLY" if s>=70
                                           else "PARTIALLY" if s>=50 else "MINIMALLY" if s>=30 else "NOT_MET")
        if "what_the_answer_demonstrates" not in result:
            result["what_the_answer_demonstrates"] = result.get("matched_knowledge_points") or result.get("evidence_items") or []
        if "what_is_missing" not in result:
            result["what_is_missing"] = result.get("missing_or_weak_knowledge_points") or result.get("gaps") or []
        if "judgement" not in result:
            result["judgement"] = "Satisfactory" if result.get("overall_score_percent",0)>=70 else "Not Satisfactory"
        if "next_step" not in result:
            s = result.get("overall_score_percent", 0)
            result["next_step"] = ("PROCEED" if s>=70 else "PROBE" if s>=50 else "INTERVIEW" if s>=30 else "NOT_DEMONSTRATED")
        if question_obj:
            result["pc_id"] = question_obj.pc_id

        await save_assessment(req.assessment_id, f"knowledge_q{req.q_num}",
            {"unit_code": req.unit_code, "question": req.question,
             "answer": req.answer, "analysis": result, "ai_detection": ai_detection,
             "timestamp": datetime.now(timezone.utc).isoformat()})

        # Also write analysis into progress.knowledge_analyses so trainer view can read it
        if req.assessment_id:
            try:
                from .database import _firestore
                db = _firestore()
                if db:
                    # Read current progress, merge analysis in, write back
                    doc = await db.collection("rpl_assessments").document(req.assessment_id).get()
                    if doc.exists:
                        prog = doc.to_dict().get("progress", {})
                        if "knowledge_analyses" not in prog:
                            prog["knowledge_analyses"] = {}
                        if req.unit_code not in prog["knowledge_analyses"]:
                            prog["knowledge_analyses"][req.unit_code] = {}
                        # Store with pc_id attached for trainer cross-reference
                        analysis_record = {**result}
                        if question_obj:
                            analysis_record["pc_id"] = question_obj.pc_id
                            analysis_record["question_text"] = question_obj.text
                        prog["knowledge_analyses"][req.unit_code][str(req.q_num - 1)] = analysis_record
                        await db.collection("rpl_assessments").document(req.assessment_id).update(
                            {"progress.knowledge_analyses": prog["knowledge_analyses"]})
            except Exception as e:
                logger.warning(f"Failed to update knowledge_analyses in progress: {e}")

        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/assessment/ai-detection-report")
async def ai_detection_report(req: EvidenceSummaryRequest,
                               user: dict = Depends(current_user)):
    """
    Full AI usage detection report across all responses in an assessment.
    Analyses knowledge responses and conversation turns for AI indicators.
    """
    assessment = await get_assessment(req.assessment_id)
    _check_record_tenant(assessment, user)
    unit = registry.get(req.unit_code)
    if not unit: raise HTTPException(404, f"Unit {req.unit_code} not found")
    try:
        report = await analyse_assessment_for_ai_usage(
            get_client(), MODEL, unit, assessment,
            assessment.get("candidate", {}))
        await save_assessment(req.assessment_id,
            f"ai_detection_{req.unit_code}", report)
        return report
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/units/{unit_code}/generate-questions")
async def generate_unit_questions(unit_code: str,
                                   industry_context: str = "",
                                   user: dict = Depends(require_admin)):
    """
    Regenerate AI-quality knowledge check questions for a unit.
    Uses the full VET instructional design framework.
    """
    unit = registry.get(unit_code)
    if not unit:
        raise HTTPException(404, f"Unit {unit_code} not found")
    try:
        questions = await generate_knowledge_questions_for_unit(
            get_client(), MODEL, unit, industry_context)
        # Update in registry
        from pathlib import Path
        import json as _json
        unit_data = unit.model_dump()
        unit_data["knowledge_questions"] = questions
        pkg_dir = Path("units") / unit.training_package.lower()
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / f"{unit_code}.json").write_text(_json.dumps(unit_data, indent=2))
        # Save to Firestore
        gcp = os.getenv("GOOGLE_CLOUD_PROJECT")
        if gcp:
            try:
                from google.cloud import firestore as _fs
                _db = _fs.Client(project=gcp)
                _db.collection("rpl_unit_registry").document(unit_code).update(
                    {"knowledge_questions": questions})
            except Exception as e:
                logger.warning(f"Firestore question update failed: {e}")
        # Reload registry
        from .unit_registry import UnitOfCompetency
        updated = UnitOfCompetency(**unit_data)
        registry.add(updated)
        return {"unit_code": unit_code, "questions_generated": len(questions),
                "questions": questions}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/mapping/generate")
async def generate_mapping(req: MappingRequest, user: dict = Depends(current_user)):
    unit = registry.get(req.unit_code)
    if not unit:
        raise HTTPException(404, f"Unit {req.unit_code} not found. "
                                 f"Import via POST /api/units/import/{req.unit_code}")
    try:
        # Load full assessment context for orchestrator (tenant-scoped)
        assessment = await get_assessment(req.assessment_id) if req.assessment_id else {}
        if req.assessment_id:
            _check_record_tenant(assessment, user)
        progress = (assessment or {}).get("progress", {})
        uploads = progress.get("uploads", req.uploads or {})
        candidate_notes = progress.get("candidate_notes", req.candidate_notes or {})
        industry_context = progress.get("industry_context", req.industry_context or "")

        if req.use_orchestrator:
            logger.info(f"Orchestrator mapping: {unit.code}")
            report = await orchestrate_rpl_assessment(
                client=get_client(),
                unit=unit,
                candidate=req.candidate,
                evidence=req.evidence_summary,
                knowledge_responses=req.knowledge_responses,
                checklist_results=req.checklist_results,
                uploads=uploads,
                candidate_notes=candidate_notes,
                industry_context=industry_context,
            )
        else:
            logger.info(f"Legacy mapping: {unit.code}")
            report = await run_mapping(
                client=get_client(), model=MODEL, unit=unit,
                candidate=req.candidate,
                evidence=req.evidence_summary,
                knowledge_responses=req.knowledge_responses,
                checklist_results=req.checklist_results)

        report["audit"] = {
            "assessment_id": req.assessment_id, "unit_code": req.unit_code,
            "unit_title": unit.title, "rto": "ABC Training RTO #5800",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": MODEL, "hitl_status": "PENDING_ASSESSOR_REVIEW",
            "zdr_confirmed": True,
            "architecture": "hierarchical_multi_agent_v1" if req.use_orchestrator else "legacy_single_agent"}
        await save_assessment(req.assessment_id, "mapping_report", report)
        return report
    except Exception as e:
        logger.error(f"Mapping error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/mapping/generate-multi-unit")
async def generate_multi_unit_mapping(data: dict,
                                       user: dict = Depends(current_user)):
    """Run the full multi-unit orchestration across all units in an assessment."""
    assessment_id = data.get("assessment_id")
    unit_codes    = data.get("unit_codes", [])
    if not assessment_id or not unit_codes:
        raise HTTPException(400, "assessment_id and unit_codes required")

    assessment = await get_assessment(assessment_id)
    _check_record_tenant(assessment, user)

    units = [registry.get(c) for c in unit_codes if registry.get(c)]
    if not units:
        raise HTTPException(404, "No valid units found")

    progress        = assessment.get("progress", {})
    candidate       = assessment.get("candidate", {})
    evidence        = progress.get("candidate_notes", {}).get("resume", "")
    k_responses     = progress.get("knowledge_responses", {})
    c_results       = progress.get("checklist", {})
    uploads         = progress.get("uploads", {})
    candidate_notes = progress.get("candidate_notes", {})
    industry_context = progress.get("industry_context", data.get("industry_context",""))

    try:
        results = await orchestrate_multi_unit_assessment(
            client=get_client(),
            units=units,
            candidate=candidate,
            evidence=evidence,
            knowledge_responses=k_responses,
            checklist_results=c_results,
            uploads=uploads,
            candidate_notes=candidate_notes,
            industry_context=industry_context,
        )
        await save_assessment(assessment_id, "multi_unit_mapping", results)
        return results
    except Exception as e:
        logger.error(f"Multi-unit mapping error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/gap-analysis/generate")
async def gap_analysis(data: dict, user: dict = Depends(current_user)):
    unit = registry.get(data.get("unit_code", ""))
    if not unit: raise HTTPException(404, "Unit not found")
    if data.get("assessment_id"):
        _check_record_tenant(await get_assessment(data["assessment_id"]), user)
    try:
        result = await run_gap_analysis(get_client(), MODEL, unit, data.get("mapping", {}))
        if data.get("assessment_id"):
            await save_assessment(data["assessment_id"], "gap_analysis", result)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/parse-document")
async def parse_doc(file: UploadFile = File(...), doc_type: str = Form(...),
                    assessment_id: str = Form(...), unit_code: str = Form(default=""),
                    token: str = Form(default="")):
    # Invite token authenticates the candidate and must own this assessment.
    await _verify_student_token(token, assessment_id)
    from .document_ai import parse_document, redact_sensitive
    content = await file.read()
    # Bound upload size — guard against memory exhaustion (10 MB).
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10 MB).")
    try:
        parsed = await parse_document(content, file.content_type, doc_type)
        parsed = redact_sensitive(parsed)
        await save_assessment(assessment_id, f"doc_{doc_type}",
            {"filename": file.filename, "doc_type": doc_type, "unit_code": unit_code,
             "token_count": parsed["token_count"]})
        return {"assessment_id": assessment_id, "filename": file.filename, "parsed": parsed}
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# FRONTEND ROUTING
# Two portals — trainer (/trainer) and student (/rpl/{token})
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# BULK CREATE — upload CSV of candidates
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/trainer/assessments/bulk-create")
async def bulk_create_assessments(
    file: UploadFile = File(...),
    trainer_name: str = Form(...),
    trainer_email: str = Form(...),
    unit_codes: str = Form(...),     # comma-separated
    notes: str = Form(default=""),
    user: dict = Depends(current_user)
):
    """Upload a CSV of candidates to create multiple assessments at once."""
    import csv, io

    codes = [c.strip().upper() for c in unit_codes.split(",") if c.strip()]
    for code in codes:
        if not registry.get(code):
            raise HTTPException(404, f"Unit {code} not found")

    content = (await file.read()).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))

    # Normalise header names — tolerate different capitalisation
    results = []
    errors  = []
    base_url = os.getenv("BASE_URL", "")
    # Identity from JWT — request body fields are ignored for security
    trainer_name  = user.get("name") or trainer_name
    trainer_email = user.get("email") or trainer_email
    trainer_id = f"trainer_{trainer_email.replace('@','_').replace('.','_')}"

    for i, row in enumerate(reader, 1):
        # Accept Name/name/FULL NAME etc.
        def col(*keys):
            for k in keys:
                for rk in row:
                    if rk.strip().lower() == k.lower():
                        return (row[rk] or "").strip()
            return ""

        name     = col("name", "full name", "candidate name", "candidate")
        email    = col("email", "email address", "candidate email")
        employer = col("employer", "company", "organisation", "organization")
        role     = col("role", "job title", "position", "title")
        duration = col("duration", "tenure", "years")

        if not name or not email:
            errors.append(f"Row {i}: missing name or email")
            continue

        try:
            assessment_id, invite_token = await create_assessment(trainer_id, {
                "org_id":          user["org_id"],
                "trainer_user_id": user["id"],
                "trainer_name":    trainer_name,
                "trainer_email":   trainer_email,
                "unit_codes":      codes,
                "candidate":       {"name": name, "email": email,
                                    "employer": employer, "role": role,
                                    "duration": duration},
                "notes": notes,
            })
            invite_url = f"{base_url}/rpl/{invite_token}"
            await _notify_candidate_invite({
                "candidate": {"name": name, "email": email},
                "trainer_name": trainer_name, "trainer_email": trainer_email,
                "unit_codes": codes, "invite_url": invite_url,
                "assessment_id": assessment_id, "notes": notes,
            })
            results.append({
                "name": name, "email": email,
                "assessment_id": assessment_id,
                "invite_url": invite_url, "status": "created"
            })
        except Exception as e:
            errors.append(f"Row {i} ({name}): {str(e)[:80]}")

    return {
        "created": len(results),
        "errors":  errors,
        "assessments": results,
        "message": f"Created {len(results)} assessments"
    }


# ══════════════════════════════════════════════════════════════════════════════
# ASSESSMENT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/trainer/templates")
async def list_templates(user: dict = Depends(current_user)):
    templates = await _load_templates()
    return {"templates": templates}

@app.post("/api/trainer/templates")
async def create_template(req: TemplateCreate,
                          user: dict = Depends(current_user)):
    import uuid
    template = {
        "id": str(uuid.uuid4())[:8],
        "name": req.name,
        "description": req.description,
        "unit_codes": req.unit_codes,
        "created_by": req.trainer_email,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    templates = await _load_templates()
    templates.append(template)
    await _save_templates(templates)
    return template

@app.delete("/api/trainer/templates/{template_id}")
async def delete_template(template_id: str,
                          user: dict = Depends(current_user)):
    templates = await _load_templates()
    templates = [t for t in templates if t["id"] != template_id]
    await _save_templates(templates)
    return {"deleted": True}

async def _load_templates() -> list:
    from pathlib import Path
    p = Path("units/_templates.json")
    if p.exists():
        return json.loads(p.read_text())
    # Try Firestore
    gcp = os.getenv("GOOGLE_CLOUD_PROJECT")
    if gcp:
        try:
            from google.cloud import firestore as _fs
            db = _fs.Client(project=gcp)
            doc = db.collection("rpl_config").document("templates").get()
            if doc.exists:
                return doc.to_dict().get("items", [])
        except Exception:
            pass
    return []

async def _save_templates(templates: list):
    from pathlib import Path
    Path("units/_templates.json").write_text(json.dumps(templates, indent=2))
    gcp = os.getenv("GOOGLE_CLOUD_PROJECT")
    if gcp:
        try:
            from google.cloud import firestore as _fs
            db = _fs.Client(project=gcp)
            db.collection("rpl_config").document("templates").set({"items": templates})
        except Exception as e:
            logger.warning(f"Template Firestore save failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL INVITE (auto-send on assessment create)
# ══════════════════════════════════════════════════════════════════════════════

async def _notify_candidate_invite(assessment_data: dict):
    """Send invite email directly to the candidate."""
    api_key = os.getenv("SENDGRID_API_KEY")
    if not api_key:
        logger.info("SendGrid not configured — skipping candidate invite email")
        return
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        candidate   = assessment_data.get("candidate", {})
        invite_url  = assessment_data.get("invite_url", "")
        trainer     = assessment_data.get("trainer_name", "Your trainer")
        unit_codes  = ", ".join(assessment_data.get("unit_codes", []))
        notes_block = (f"<p><strong>Note from your trainer:</strong> {assessment_data['notes']}</p>"
                       if assessment_data.get("notes") else "")
        msg = Mail(
            from_email=os.getenv("SENDGRID_FROM", "rpl@abctraining.com.au"),
            to_emails=candidate.get("email"),
            subject=f"Your RPL assessment is ready — {unit_codes}",
            html_content=f"""
            <p>Hi {candidate.get('name', 'there')},</p>
            <p>{trainer} at <strong>ABC Training RTO #5800</strong> has set up an RPL assessment for you.</p>
            <p><strong>Units:</strong> {unit_codes}</p>
            {notes_block}
            <p>Click the link below to begin your RPL. You can save your progress at any time and return later.</p>
            <p><a href="{invite_url}" style="background:#1F2060;color:#fff;padding:12px 24px;border-radius:8px;text-decoration:none;display:inline-block">Begin your RPL assessment →</a></p>
            <p>Or copy this link: {invite_url}</p>
            <p>If you have questions, contact {trainer} directly.</p>
            <p>ABC Training | RTO #5800</p>""")
        sg = SendGridAPIClient(api_key)
        sg.send(msg)
        logger.info(f"Invite sent to {candidate.get('email')}")
    except Exception as e:
        logger.warning(f"Candidate invite email failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# AI QUALITY — Cross-unit mapping + Third-party report
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/mapping/cross-unit")
async def cross_unit_mapping(req: CrossUnitRequest, user: dict = Depends(current_user)):
    if req.assessment_id:
        _check_record_tenant(await get_assessment(req.assessment_id), user)
    units = [registry.get(c) for c in req.unit_codes if registry.get(c)]
    if len(units) < 2:
        raise HTTPException(400, "Cross-unit mapping requires at least 2 valid units")
    try:
        result = await run_cross_unit_mapping(
            get_client(), MODEL, units, req.candidate,
            req.evidence_summary, req.knowledge_responses)
        if req.assessment_id:
            await save_assessment(req.assessment_id, "cross_unit_mapping", result)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/mapping/third-party-report")
async def third_party_report(req: ThirdPartyReportRequest, user: dict = Depends(current_user)):
    if req.assessment_id:
        _check_record_tenant(await get_assessment(req.assessment_id), user)
    unit = registry.get(req.unit_code)
    if not unit:
        raise HTTPException(404, f"Unit {req.unit_code} not found")
    try:
        result = await generate_third_party_report_template(
            get_client(), MODEL, unit, req.mapping, req.candidate)
        if req.assessment_id:
            await save_assessment(req.assessment_id,
                f"third_party_template_{req.unit_code}", result)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# COMPETENCY CONVERSATION — embedded in student portal
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/conversation/start")
async def conversation_start(data: dict):
    """Initialise a competency conversation session for a student."""
    assessment_id = data.get("assessment_id")
    unit_code     = data.get("unit_code")
    # Invite token authenticates the candidate and must own this assessment.
    assessment    = await _verify_student_token(data.get("token", ""), assessment_id)
    unit          = registry.get(unit_code) if unit_code else None

    if not unit:
        raise HTTPException(404, f"Unit {unit_code} not found")

    # Pull gap questions from existing mapping if available
    gap_questions = []
    if assessment:
        mapping = assessment.get("progress", {}).get("mapping")
        if mapping:
            for el in mapping.get("elements", []):
                for pc in el.get("pcs", []):
                    if pc.get("verdict") in ("PARTIAL", "GAP") and pc.get("followup"):
                        gap_questions.append({
                            "pc":       pc["id"],
                            "text":     pc["followup"],
                            "element":  el.get("title", ""),
                            "verdict":  pc["verdict"],
                            "confidence": pc.get("confidence", 0)
                        })

    # Fall back to knowledge questions if no mapping yet
    if not gap_questions:
        gap_questions = [
            {"pc": q["pc_refs"][0] if q["pc_refs"] else q["element_ref"],
             "text": q["text"], "element": q["element_ref"],
             "verdict": "UNKNOWN", "confidence": 0}
            for q in unit.knowledge_questions[:8]
        ]

    session = {
        "unit_code":      unit_code,
        "unit_title":     unit.title,
        "assessment_id":  assessment_id,
        "candidate_name": (assessment or {}).get("candidate", {}).get("name", "Candidate"),
        "questions":      gap_questions,
        "total":          len(gap_questions),
        "session_id":     f"{assessment_id}_{unit_code}_{int(__import__('time').time())}"
    }
    return session


@app.post("/api/conversation/analyse")
async def conversation_analyse(data: dict):
    """Analyse a single conversation response."""
    unit_code     = data.get("unit_code")
    question      = data.get("question")
    answer        = data.get("answer", "")
    pc_refs       = data.get("pc_refs", [])
    element_ref   = data.get("element_ref", "")
    assessment_id = data.get("assessment_id", "")

    # Invite token authenticates the candidate and must own this assessment.
    await _verify_student_token(data.get("token", ""), assessment_id or None)

    unit = registry.get(unit_code)
    if not unit:
        raise HTTPException(404, f"Unit {unit_code} not found")
    if len((answer or "").split()) < 5:
        raise HTTPException(400, "Response too short")

    result = await analyse_knowledge_response(
        client=get_client(), model=MODEL, unit=unit,
        question=question, answer=answer,
        pc_refs=pc_refs, element_ref=element_ref)

    if assessment_id:
        await save_assessment(assessment_id, f"conv_{pc_refs[0] if pc_refs else 'q'}",
            {"question": question, "answer": answer, "analysis": result,
             "timestamp": datetime.now(timezone.utc).isoformat()})
    return result


# ══════════════════════════════════════════════════════════════════════════════
# GUIDED CONVERSATION TURN
# Takes the full dialogue history + benchmark and generates a contextual
# follow-up question targeting exactly what is still missing
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/conversation/guided-turn")
async def guided_conversation_turn(data: dict):
    """
    Analyse a candidate's response in the context of the full dialogue so far.
    Returns: analysis of this turn + a targeted follow-up if more is needed.
    """
    unit_code        = data.get("unit_code", "")
    pc_id            = data.get("pc_id", "")
    benchmark        = data.get("benchmark_statement", "")
    analysis_prompt  = data.get("analysis_prompt", "")
    dialogue_history = data.get("dialogue_history", [])  # [{role, content, turn}]
    latest_answer    = data.get("latest_answer", "")
    turn_number      = data.get("turn_number", 1)
    assessment_id    = data.get("assessment_id", "")
    max_turns        = data.get("max_turns", 4)

    # Invite token authenticates the candidate and must own this assessment.
    await _verify_student_token(data.get("token", ""), assessment_id or None)

    unit = registry.get(unit_code)
    if not unit:
        raise HTTPException(404, f"Unit {unit_code} not found")

    wc = len((latest_answer or "").split())

    # Build the full conversation context for the prompt
    history_text = ""
    for turn in dialogue_history:
        role = turn.get("role", "")
        tcontent = turn.get("content", "")
        if role == "assessor":
            history_text += f"\nAssessor asked: {tcontent}"
        elif role == "candidate":
            history_text += f"\nCandidate responded: {tcontent}"

    # Detect template benchmark — same logic as knowledge prompt
    pc_text_raw = next((pc.text for el in unit.elements
                        for pc in el.pcs if pc.id == pc_id), '')
    benchmark_is_template = (
        not benchmark or
        benchmark.lower().strip().rstrip('.') ==
        f"the candidate demonstrates that they can {pc_text_raw.lower().strip().rstrip('.')}"
    )

    if benchmark_is_template:
        scoring_instruction = f"""IMPORTANT: The benchmark for this PC is a template restatement.
Do NOT match against benchmark wording. Score on SPECIFICITY and TECHNICAL QUALITY.

SPECIFICITY TEST — apply first:
FAIL (cap score at 0.45) if:
  - No specific employer, workplace, or organisation named
  - Uses third-person ("a worker must", "one should") instead of first-person ("I", "we")
  - Structured with headers/numbered sections not asked for
  - Could be written by someone who has never worked in this field

PASS (can score 0.70+) only if:
  - Names specific employer/organisation
  - References specific products, equipment, or procedures by name
  - First-person voice describing actual experience

If FAILS specificity test:
  Detailed generic answer → 0.25–0.42
  Some relevant content → 0.20–0.35
  Vague one-liners → 0.05–0.20

If PASSES specificity test, count concrete workplace-specific claims:
  8+ specific claims → 0.80–0.92
  5–7 claims → 0.68–0.79
  3–4 claims → 0.52–0.67
  1–2 claims → 0.35–0.51

NEVER score above 0.45 if no employer or specific workplace is named.
The PC this conversation covers: {pc_text_raw}"""
    else:
        scoring_instruction = f"""Compare the candidate's response directly against this benchmark:
BENCHMARK: {benchmark}

VACS requirement: {analysis_prompt}

Score 0.70+ only if the response substantially meets the benchmark.
Score 0.50 only for genuinely borderline responses.
NEVER score 0.50 for a technically detailed response — that would mean it equals a one-liner."""

    # Load candidate context for sector-appropriate questions
    assessment_rec = await get_assessment(assessment_id) if assessment_id else {}
    cand = (assessment_rec or {}).get("candidate", {})
    cand_role     = cand.get("role", "")
    cand_employer = cand.get("employer", "")
    cand_industry = cand.get("industry", "") or cand.get("industry_context", "")
    cand_line = f"Candidate: {cand_role} at {cand_employer}" if cand_employer else ""
    sector_line = f"Their sector: {cand_industry}." if cand_industry else ""

    system = f"""You are a skilled Australian VET assessor conducting a guided competency conversation for {unit_code} — {unit.title}.

{cand_line}
{sector_line}
Your role is to GUIDE the candidate to demonstrate competency through a natural dialogue.
You do NOT simply accept or reject — you ask follow-up questions that help the candidate provide the specific evidence needed.

SECTOR ALIGNMENT: Ask questions that are relevant to the candidate's actual sector.
If they work in civil construction, probe for civil construction scenarios.
If they work in laboratory science, probe for lab-specific procedures.
Reference their employer ({cand_employer or "their workplace"}) and role ({cand_role or "their position"}) when asking follow-ups.
Do NOT ask questions drawn from unrelated industries.

{scoring_instruction}

GUIDING PRINCIPLES:
1. Analyse what the candidate HAS provided — count specific technical claims
2. Identify SPECIFICALLY what is still missing or could be strengthened
3. If turns remain and evidence could be stronger: generate ONE focused follow-up
4. The follow-up must reference their specific workplace context where possible
5. Do NOT ask about things already covered in the dialogue
6. Use plain, encouraging language — this is a conversation, not an interrogation
7. If the response is technically strong (confidence >= 0.70): declare Satisfactory

Turn {turn_number} of {max_turns} maximum.
Respond ONLY in valid JSON."""

    user = f"""PC {pc_id}: {pc_text_raw}
Candidate role/employer: {cand_role or "unknown"} at {cand_employer or "their workplace"}

{('Benchmark: ' + benchmark) if not benchmark_is_template else 'Scoring: Count concrete technical claims relevant to their sector (see system prompt)'}

Full conversation so far:
{history_text}

Candidate's latest response (turn {turn_number}): "{latest_answer}"
Word count: {wc}
{"ALERT: Very short — probe for more detail" if wc < 20 else ""}

{"Count concrete technical claims relevant to their sector. Generic claims score lower than sector-specific ones." if benchmark_is_template else "Analyse against the benchmark above."}
If generating a follow-up question, frame it around their actual workplace: {cand_employer or "their current workplace"}.

Return JSON:
{{
  "turn_analysis": {{
    "confidence": 0.0-1.0,
    "judgement": "Satisfactory"|"Not Satisfactory",
    "what_was_demonstrated": ["specific technical claim or domain knowledge from THIS response"],
    "what_is_still_missing": ["specific gap — what would strengthen this response"],
    "evidence_items": ["direct quote from response showing technical knowledge"]
  }},
  "cumulative_analysis": {{
    "confidence": 0.0-1.0,
    "judgement": "Satisfactory"|"Not Satisfactory",
    "summary": "How many concrete technical claims total across the dialogue — why this confidence score",
    "evidence_items": ["key technical claim from the dialogue"],
    "gap_notes": "What specific knowledge is still missing — empty if Satisfactory"
  }},
  "next_action": "FOLLOW_UP"|"SATISFIED"|"MAX_TURNS_REACHED",
  "follow_up_question": "ONE targeted question for the most important gap — empty if SATISFIED or MAX_TURNS_REACHED",
  "encouragement": "One sentence acknowledging what the candidate got right technically",
  "transition_summary": "If SATISFIED or MAX_TURNS_REACHED: what the candidate demonstrated"
}}"""

    try:
        client = get_client()
        loop = asyncio.get_event_loop()
        def _call():
            return client.messages.create(
                model=MODEL, max_tokens=1000,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
        response = await loop.run_in_executor(None, _call)
        raw = response.content[0].text.strip().replace("```json","").replace("```","").strip()
        result = json.loads(raw)

        # ── Safety override: enforce follow-up logic based on actual confidence ──
        cumulative    = result.get("cumulative_analysis", {})
        cum_confidence = cumulative.get("confidence", 0)
        current_action = result.get("next_action", "FOLLOW_UP")

        # If confidence < 0.70 and turns remain, force a follow-up
        if cum_confidence < 0.70 and turn_number < max_turns:
            result["next_action"] = "FOLLOW_UP"
            # If Claude didn't provide a follow-up question, generate a targeted one
            if not result.get("follow_up_question"):
                gap_notes = cumulative.get("gap_notes", "")
                missing   = result.get("turn_analysis", {}).get("what_is_still_missing", [])
                missing_text = gap_notes or (missing[0] if missing else "")
                if missing_text:
                    result["follow_up_question"] = (
                        f"You've made a good start. Can you give me a specific workplace example "
                        f"where you had to {missing_text.lower().rstrip('.')}? "
                        f"What exactly did you do and what was the outcome?"
                    )
                else:
                    result["follow_up_question"] = (
                        f"That's helpful. Can you give me a concrete example from your current "
                        f"workplace — what specific steps did you take, and what was the result?"
                    )

        # If confidence >= 0.70, mark as satisfied regardless of what Claude said
        elif cum_confidence >= 0.70 and current_action == "FOLLOW_UP":
            result["next_action"] = "SATISFIED"
            if not result.get("transition_summary"):
                result["transition_summary"] = (
                    f"Good response — you've demonstrated solid knowledge of this topic. "
                    f"Moving to the next question."
                )

        # Max turns reached — always close
        if turn_number >= max_turns:
            result["next_action"] = "MAX_TURNS_REACHED"

        # Save to assessment
        if assessment_id:
            await save_assessment(assessment_id, f"conv_turn_{unit_code}_{pc_id}_{turn_number}", {
                "unit_code": unit_code, "pc_id": pc_id, "turn": turn_number,
                "answer": latest_answer, "analysis": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        return result
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 3 — Annotated Evidence Portfolio Summary
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/assessment/evidence-summary")
async def evidence_summary(req: EvidenceSummaryRequest,
                           user: dict = Depends(current_user)):
    assessment = await get_assessment(req.assessment_id)
    _check_record_tenant(assessment, user)
    unit = registry.get(req.unit_code)
    if not unit:
        raise HTTPException(404, f"Unit {req.unit_code} not found")
    try:
        result = await generate_evidence_portfolio_summary(
            get_client(), MODEL, unit, assessment, req.industry_context)
        await save_assessment(req.assessment_id, f"evidence_summary_{req.unit_code}", result)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 4 — Benchmark Gap Report
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/assessment/gap-report")
async def benchmark_gap_report(req: EvidenceSummaryRequest,
                                user: dict = Depends(current_user)):
    assessment = await get_assessment(req.assessment_id)
    _check_record_tenant(assessment, user)
    unit = registry.get(req.unit_code)
    if not unit:
        raise HTTPException(404, f"Unit {req.unit_code} not found")
    mapping = assessment.get("progress", {}).get("mapping", {})
    if not mapping:
        raise HTTPException(400, "No mapping found — run AI mapping first")
    try:
        result = await generate_benchmark_gap_report(
            get_client(), MODEL, unit, mapping,
            assessment.get("candidate", {}), req.industry_context)
        await save_assessment(req.assessment_id, f"gap_report_{req.unit_code}", result)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 5 — Historical Assessment Patterns
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/assessment/patterns/{unit_code}")
async def assessment_patterns(unit_code: str,
                               user: dict = Depends(current_user)):
    unit = registry.get(unit_code)
    if not unit:
        raise HTTPException(404, f"Unit {unit_code} not found")
    # Fetch all completed assessments for this unit
    all_assessments = await list_assessments(status="COMPLETE", limit=100)
    unit_assessments = [
        a for a in all_assessments
        if unit_code in a.get("unit_codes", [])
    ]
    if len(unit_assessments) < 2:
        return {"message": f"Only {len(unit_assessments)} completed assessment(s) for {unit_code} — need at least 2 for pattern analysis",
                "count": len(unit_assessments)}
    try:
        result = await generate_assessment_patterns(
            get_client(), MODEL, unit_code, unit.title, unit_assessments)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 6 — Industry Context
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/assessment/industry-context")
async def industry_context(req: IndustryContextRequest,
                            user: dict = Depends(current_user)):
    assessment = await get_assessment(req.assessment_id)
    _check_record_tenant(assessment, user)
    # Save context to assessment
    progress = assessment.get("progress", {})
    progress["industry_context"] = req.industry_context
    progress["industry_sector"]  = req.industry_sector
    await save_progress(req.assessment_id, progress)

    # Generate context profile for each unit
    results = {}
    for unit_code in assessment.get("unit_codes", []):
        unit = registry.get(unit_code)
        if unit:
            try:
                result = await generate_industry_context_profile(
                    get_client(), MODEL, unit,
                    assessment.get("candidate", {}),
                    req.industry_context, req.industry_sector)
                results[unit_code] = result
                await save_assessment(req.assessment_id,
                    f"industry_context_{unit_code}", result)
            except Exception as e:
                results[unit_code] = {"error": str(e)}
    return {"industry_context": req.industry_context, "unit_profiles": results}


@app.get("/api/assessment/{assessment_id}/industry-context")
async def get_industry_context(assessment_id: str,
                                user: dict = Depends(current_user)):
    assessment = await get_assessment(assessment_id)
    _check_record_tenant(assessment, user)
    progress = assessment.get("progress", {})
    return {
        "industry_context": progress.get("industry_context", ""),
        "industry_sector":  progress.get("industry_sector", "")
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 7 — Determination Worksheet
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/assessment/determination-worksheet")
async def determination_worksheet(req: EvidenceSummaryRequest,
                                   user: dict = Depends(current_user)):
    assessment = await get_assessment(req.assessment_id)
    _check_record_tenant(assessment, user)
    unit = registry.get(req.unit_code)
    if not unit:
        raise HTTPException(404, f"Unit {req.unit_code} not found")
    try:
        result = await generate_determination_worksheet(
            get_client(), MODEL, unit, assessment, req.industry_context)
        await save_assessment(req.assessment_id,
            f"determination_worksheet_{req.unit_code}", result)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/assessment/determination-submit")
async def submit_determination(req: DeterminationRequest,
                                user: dict = Depends(current_user)):
    """Save the trainer's completed determination — the formal ASQA-compliant record."""
    determination = {
        "assessment_id":          req.assessment_id,
        "unit_code":              req.unit_code,
        "industry_context":       req.industry_context,
        "pc_determinations":      req.pc_determinations,
        "overall_determination":  req.overall_determination,
        "assessor_rationale":     req.assessor_rationale,
        "reasonable_adjustments": req.reasonable_adjustments,
        "assessor_name":          req.assessor_name,
        "assessor_id":            req.assessor_id,
        "determined_at":          datetime.now(timezone.utc).isoformat(),
        "rto":                    "ABC Training RTO #5800",
        "hitl_compliant":         True,
        "source":                 "HUMAN_ASSESSOR",
    }
    await save_assessment(req.assessment_id,
        f"formal_determination_{req.unit_code}", determination)
    if req.overall_determination == "RPL Granted":
        await complete_assessment(req.assessment_id)
    return {"saved": True, "determination": determination}


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 10 — Pre-Assessment Screening
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/assessment/pre-screen")
async def pre_screen(req: PreScreenRequest,
                     user: dict = Depends(current_user)):
    units = [registry.get(c) for c in req.unit_codes if registry.get(c)]
    if not units:
        raise HTTPException(404, "No valid units found")
    try:
        result = await run_pre_assessment_screen(
            get_client(), MODEL, units, req.candidate,
            req.resume_text, req.industry_context)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# PRIVACY — APP 5 Collection Notice, APP 12 Access, APP 13 Correction
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/privacy/acknowledge")
async def privacy_acknowledge(data: dict):
    """
    Record that the candidate has read and acknowledged the Privacy Collection
    Notice before any personal data is loaded. Required under APP 5.
    """
    token        = data.get("token", "")
    acknowledged = data.get("acknowledged", False)
    timestamp    = datetime.now(timezone.utc).isoformat()

    if not token or not acknowledged:
        raise HTTPException(400, "Token and acknowledged=true required")

    # Look up assessment to get ID
    assessment = await get_by_token(token)
    if not assessment:
        # Token not found — record consent attempt but don't block the flow
        logger.warning(f"Privacy acknowledge: token {token[:8]}... not found in DB")
        return {"recorded": False, "timestamp": timestamp,
                "note": "Token not found — consent noted but not linked to assessment"}

    assessment_id = assessment["assessment_id"]

    # Record consent
    consent_record = {
        "token":         token,
        "acknowledged":  True,
        "timestamp":     timestamp,
        "ip_hash":       "redacted",   # do not store raw IP — hash only if needed
        "notice_version": "1.0",
        "rto":           "ABC Training RTO #5800",
        "apps_covered":  ["APP3","APP5","APP6","APP11","APP12","APP13"],
    }
    await save_assessment(assessment_id, "privacy_consent", consent_record)

    # Save to Firestore privacy_consents collection for audit trail
    gcp = os.getenv("GOOGLE_CLOUD_PROJECT")
    if gcp:
        try:
            from google.cloud import firestore as _fs
            db = _fs.Client(project=gcp)
            db.collection("privacy_consents").document(assessment_id).set(consent_record)
        except Exception as e:
            logger.warning(f"Privacy consent Firestore save failed: {e}")

    return {"recorded": True, "timestamp": timestamp, "assessment_id": assessment_id}


@app.get("/api/privacy/check/{token}")
async def privacy_check(token: str):
    """Check whether this candidate has already acknowledged the notice.
    Always returns 200 — never 404. Unknown tokens return acknowledged:false."""
    try:
        assessment = await get_by_token(token)
        if not assessment:
            return {"acknowledged": False}
        gcp = os.getenv("GOOGLE_CLOUD_PROJECT")
        if gcp:
            try:
                from google.cloud import firestore as _fs
                db = _fs.Client(project=gcp)
                doc = db.collection("privacy_consents").document(
                    assessment["assessment_id"]).get()
                if doc.exists:
                    return {"acknowledged": True,
                            "timestamp": doc.to_dict().get("timestamp")}
            except Exception:
                pass
        return {"acknowledged": False}
    except Exception:
        return {"acknowledged": False}


@app.get("/api/candidate/my-data/{token}")
async def candidate_my_data(token: str):
    """
    APP 12 — Candidate access to their own data.
    Returns all information held about this candidate in structured form.
    """
    assessment = await get_by_token(token)
    if not assessment:
        raise HTTPException(404, "Invalid or expired link")

    candidate  = assessment.get("candidate", {})
    progress   = assessment.get("progress", {})

    # Build a human-readable data export
    return {
        "your_data": {
            "personal_information": {
                "name":     candidate.get("name"),
                "email":    candidate.get("email"),
                "employer": candidate.get("employer"),
                "role":     candidate.get("role"),
                "duration": candidate.get("duration"),
            },
            "assessment_information": {
                "assessment_id":   assessment.get("assessment_id"),
                "units":           assessment.get("unit_codes", []),
                "status":          assessment.get("status"),
                "created_at":      assessment.get("created_at"),
                "submitted_at":    assessment.get("submitted_at"),
                "rto":             "ABC Training RTO #5800",
            },
            "responses_provided": {
                "self_assessment_checklist": bool(progress.get("checklist")),
                "knowledge_responses":       bool(progress.get("knowledge_responses")),
                "conversation_records":      len(progress.get("conversation_records") or []),
                "documents_uploaded":        list((progress.get("uploads") or {}).keys()),
                "candidate_notes":           bool(progress.get("candidate_notes")),
            },
            "ai_analysis_records": {
                "mapping_generated":         bool(progress.get("mapping")),
                "gap_analysis_generated":    bool(progress.get("gap_analysis")),
                "knowledge_analyses":        bool(progress.get("knowledge_analyses")),
                "note": "AI analysis is used to assist your assessor only. "
                        "Your assessor makes all final decisions.",
            },
            "data_handling": {
                "storage_location":  "Australia (Google Cloud Sydney)",
                "retention_period":  "2 years from assessment completion (ASQA requirement)",
                "ai_processing":     "Zero Data Retention — Anthropic does not retain your data",
                "third_parties":     "ABC Training RTO #5800, your nominated assessor, "
                                     "ASQA on regulatory audit only",
            },
        },
        "your_rights": {
            "access":     "You are viewing your data now under APP 12",
            "correction": "Contact your trainer to correct any inaccurate information",
            "deletion":   "POST /api/candidate/anonymise/{token} to request anonymisation "
                          "(audit records required by law are retained)",
            "complaints": "Office of the Australian Information Commissioner — oaic.gov.au",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/candidate/anonymise/{token}")
async def candidate_anonymise(token: str, data: dict = {}):
    """
    APP 13 — Candidate can request anonymisation of identifying data.
    Retains competency outcome and audit trail (required by ASQA).
    Replaces PII with pseudonyms.
    """
    assessment = await get_by_token(token)
    if not assessment:
        raise HTTPException(404, "Invalid or expired link")

    # Only allow if assessment is complete
    if assessment.get("status") not in ("COMPLETE", "SUBMITTED"):
        raise HTTPException(400,
            "Anonymisation is only available after assessment completion")

    assessment_id = assessment["assessment_id"]
    candidate     = assessment.get("candidate", {})

    # Replace PII with pseudonyms — retain enough for audit
    pseudonym = f"Candidate-{assessment_id[:8]}"
    anonymised_candidate = {
        "name":     pseudonym,
        "email":    "anonymised@redacted",
        "employer": "Employer-redacted",
        "role":     candidate.get("role", ""),    # keep role — not identifying
        "duration": candidate.get("duration", ""), # keep duration — not identifying
    }

    # Update Firestore
    gcp = os.getenv("GOOGLE_CLOUD_PROJECT")
    if gcp:
        try:
            from google.cloud import firestore as _fs
            db = _fs.Client(project=gcp)
            db.collection("rpl_assessments").document(assessment_id).update({
                "candidate":      anonymised_candidate,
                "anonymised_at":  datetime.now(timezone.utc).isoformat(),
                "anonymised_by":  "candidate_request",
            })
        except Exception as e:
            raise HTTPException(500, f"Anonymisation failed: {e}")

    await save_assessment(assessment_id, "anonymisation_record", {
        "requested_at":    datetime.now(timezone.utc).isoformat(),
        "requested_by":    "candidate",
        "original_name":   candidate.get("name", ""),  # kept in sub-record for audit
        "pseudonym":       pseudonym,
        "retention_note":  "Competency outcome and audit trail retained per ASQA requirements",
    })

    return {
        "anonymised": True,
        "pseudonym":  pseudonym,
        "note": "Your identifying information has been replaced. "
                "Your competency outcome and assessment records are retained "
                "as required by ASQA for 2 years.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE PORTFOLIO REVIEW — Trainer ticks, comments, and builds portfolio
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/assessment/{assessment_id}/portfolio-review")
async def save_portfolio_review(assessment_id: str, data: dict,
                                 user: dict = Depends(current_user)):
    """
    Save the trainer's review of each evidence item.
    This is the formal HITL record — trainer confirms they have reviewed
    every piece of evidence before making a determination.
    """
    review = {
        "assessment_id":     assessment_id,
        "reviewer":          data.get("reviewer_name", "Trainer"),
        "reviewed_at":       datetime.now(timezone.utc).isoformat(),
        "rto":               "ABC Training RTO #5800",
        "evidence_reviews":  data.get("evidence_reviews", []),
        # [{
        #   evidence_id, evidence_type, evidence_label,
        #   reviewed: true/false,
        #   authenticity_confirmed: true/false,
        #   currency_confirmed: true/false,
        #   relevance_confirmed: true/false,
        #   assessor_comment: str,
        #   vacs_concern: str
        # }]
        "knowledge_reviews":     data.get("knowledge_reviews", []),
        # [{q_idx, question, answer_summary, reviewed, sufficient, assessor_comment}]
        "conversation_reviews":  data.get("conversation_reviews", []),
        # [{pc_id, reviewed, credible, assessor_comment}]
        "overall_portfolio_comment": data.get("overall_portfolio_comment", ""),
        "portfolio_sufficient":  data.get("portfolio_sufficient"),  # true/false/null
        "hitl_confirmed":        data.get("hitl_confirmed", False),
        "hitl_declaration":      (
            "I confirm I have reviewed all evidence submitted by the candidate "
            "and applied my professional judgement as a qualified assessor in "
            "accordance with Standards for RTOs 2015."
            if data.get("hitl_confirmed") else ""
        ),
    }

    await save_assessment(assessment_id, "portfolio_review", review)
    return {"saved": True, "reviewed_at": review["reviewed_at"]}


@app.get("/api/assessment/{assessment_id}/portfolio-review")
async def get_portfolio_review(assessment_id: str,
                                user: dict = Depends(current_user)):
    """Load a saved portfolio review for an assessment."""
    assessment = await get_assessment(assessment_id)
    _check_record_tenant(assessment, user)
    # Try sub-collection first
    db_obj = None
    from .database import _firestore
    db = _firestore()
    if db:
        try:
            doc = await db.collection("rpl_assessments").document(assessment_id)                         .collection("records").document("portfolio_review").get()
            if doc.exists:
                return doc.to_dict()
        except Exception:
            pass
    return {"assessment_id": assessment_id, "evidence_reviews": [],
            "knowledge_reviews": [], "conversation_reviews": [],
            "overall_portfolio_comment": "", "portfolio_sufficient": None,
            "hitl_confirmed": False}


@app.post("/api/assessment/{assessment_id}/rebuild-analyses")
async def rebuild_analyses(assessment_id: str,
                            user: dict = Depends(current_user)):
    """
    Reconstruct knowledge_analyses in progress from saved sub-records.
    Fixes assessments where analyses were saved to sub-records but not
    written back to progress.knowledge_analyses.
    """
    from .database import _firestore
    db = _firestore()
    if not db:
        raise HTTPException(503, "Firestore not available")

    rebuilt = {}
    try:
        # Scan all knowledge_q sub-records
        records_ref = db.collection("rpl_assessments").document(assessment_id).collection("records")
        async for rec_doc in records_ref.stream():
            if rec_doc.id.startswith("knowledge_q"):
                data = rec_doc.to_dict()
                unit_code = data.get("unit_code", "")
                analysis  = data.get("analysis", {})
                q_num     = data.get("analysis", {}).get("q_num") or int(rec_doc.id.replace("knowledge_q",""))
                q_idx     = str(q_num - 1)
                if unit_code and analysis:
                    if unit_code not in rebuilt:
                        rebuilt[unit_code] = {}
                    rebuilt[unit_code][q_idx] = analysis

        if rebuilt:
            await db.collection("rpl_assessments").document(assessment_id).update(
                {"progress.knowledge_analyses": rebuilt})

        return {"rebuilt": True, "units": list(rebuilt.keys()),
                "total_analyses": sum(len(v) for v in rebuilt.values())}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/assessment/{assessment_id}/analyse-question")
async def trainer_analyse_question(assessment_id: str, data: dict,
                                    user: dict = Depends(current_user)):
    """
    Trainer-triggered analysis of a specific knowledge question response.
    Runs the same analysis as the student endpoint but initiated by the trainer.
    """
    unit_code = data.get("unit_code","")
    q_idx     = int(data.get("q_idx", 0))

    assessment = await get_assessment(assessment_id)
    _check_record_tenant(assessment, user)

    unit = registry.get(unit_code)
    if not unit: raise HTTPException(404, f"Unit {unit_code} not found")

    progress = assessment.get("progress",{})
    responses = progress.get("knowledge_responses",{})
    unit_responses = responses.get(unit_code, responses) if isinstance(responses, dict) else {}
    answer = unit_responses.get(str(q_idx),"")
    if not answer:
        raise HTTPException(400, f"No answer found for Q{q_idx+1}")

    # Get question object
    question_obj = None
    if q_idx < len(unit.knowledge_questions):
        question_obj = unit.knowledge_questions[q_idx]

    candidate_data = assessment.get("candidate", {})

    def _is_placeholder(q):
        if not q: return True
        pts = q.model_answer_guide.expected_knowledge_points
        return not pts or (len(pts)==1 and pts[0].lower().startswith("technical understanding of"))

    try:
        if not _is_placeholder(question_obj):
            result = await evaluate_knowledge_answer_detailed(
                get_client(), MODEL, unit,
                question_obj.model_dump(), answer,
                candidate=candidate_data)
        else:
            result = await analyse_knowledge_response(
                get_client(), MODEL, unit,
                question_obj.text if question_obj else "",
                answer,
                [question_obj.pc_id] if question_obj else [],
                question_obj.element_ref if question_obj else "",
                candidate=candidate_data)

        # Normalise result — ensure both old and new schema fields are present
        # Old schema: confidence_score_percent (0-100)
        # New schema: overall_score_percent (0-100), commentary, meets_requirement
        if "confidence_score_percent" in result and "overall_score_percent" not in result:
            result["overall_score_percent"] = result["confidence_score_percent"]
        if "overall_score_percent" not in result:
            result["overall_score_percent"] = 0
        if "commentary" not in result:
            result["commentary"] = (result.get("evaluation_summary") or
                                    result.get("summary") or "")
        if "meets_requirement" not in result:
            score = result.get("overall_score_percent", 0)
            result["meets_requirement"] = (
                "FULLY" if score >= 85 else
                "SUBSTANTIALLY" if score >= 70 else
                "PARTIALLY" if score >= 50 else
                "MINIMALLY" if score >= 30 else "NOT_MET")
        if "what_the_answer_demonstrates" not in result:
            result["what_the_answer_demonstrates"] = (
                result.get("matched_knowledge_points") or
                result.get("evidence_items") or [])
        if "what_is_missing" not in result:
            result["what_is_missing"] = (
                result.get("missing_or_weak_knowledge_points") or
                result.get("gaps") or [])
        if "judgement" not in result:
            result["judgement"] = (
                "Satisfactory" if result.get("overall_score_percent",0) >= 70
                else "Not Satisfactory")
        if "next_step" not in result:
            score = result.get("overall_score_percent", 0)
            result["next_step"] = (
                "PROCEED" if score >= 70 else
                "PROBE" if score >= 50 else
                "INTERVIEW" if score >= 30 else "NOT_DEMONSTRATED")

        # Attach metadata
        if question_obj:
            result["pc_id"]        = question_obj.pc_id
            result["question_text"]= question_obj.text

        # Save to sub-record
        await save_assessment(assessment_id, f"knowledge_q{q_idx+1}", {
            "unit_code": unit_code, "question": question_obj.text if question_obj else "",
            "answer": answer, "analysis": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Write into progress.knowledge_analyses
        from .database import _firestore
        db = _firestore()
        if db:
            doc = await db.collection("rpl_assessments").document(assessment_id).get()
            if doc.exists:
                prog = doc.to_dict().get("progress",{})
                if "knowledge_analyses" not in prog: prog["knowledge_analyses"] = {}
                if unit_code not in prog["knowledge_analyses"]: prog["knowledge_analyses"][unit_code] = {}
                prog["knowledge_analyses"][unit_code][str(q_idx)] = result
                await db.collection("rpl_assessments").document(assessment_id).update(
                    {"progress.knowledge_analyses": prog["knowledge_analyses"]})

        return result
    except Exception as e:
        import traceback
        logger.error(f"trainer_analyse_question failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to trainer portal."""
    return RedirectResponse("/trainer")

@app.get("/trainer", response_class=HTMLResponse)
@app.get("/trainer/{path:path}", response_class=HTMLResponse)
async def trainer_portal(path: str = ""):
    with open("frontend/templates/trainer.html") as f:
        return f.read()

@app.get("/rpl/{token}", response_class=HTMLResponse)
async def student_portal(token: str):
    with open("frontend/templates/student.html") as f:
        return f.read().replace("__TOKEN__", token)

# Only mount the static directory if it exists — StaticFiles raises at import
# time on a missing directory, which would crash the whole app on boot.
if os.path.isdir("frontend/static"):
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
else:
    logger.info("frontend/static not present — skipping static mount")
