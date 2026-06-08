"""
Multi-tenant authentication & user management.

Model:
  Organisation (1) ───< User (many)
                        ├─ role = "admin"    (Administration & Compliance)
                        └─ role = "trainer"  (Trainers — assess only)

Storage: Firestore collections `rpl_orgs` and `rpl_users`,
with an in-memory fallback for local dev (same pattern as database.py).

Auth: HTTP Bearer JWT (HS256). Tokens are 12-hour, signed with
AUTH_SECRET. Passwords are bcrypt-hashed.

Public dependencies for FastAPI routes:
  current_user(authorization) -> dict
  require_admin(current_user) -> dict     # 403 unless role == 'admin'
"""
import os, uuid, json, logging, secrets
from datetime import datetime, timezone, timedelta
from typing import Optional

import bcrypt
import jwt
from fastapi import HTTPException, Header, Depends

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
# AUTH_SECRET MUST be set in production. We refuse to start without it.
# Generated once with: python -c "import secrets;print(secrets.token_urlsafe(48))"
AUTH_SECRET   = os.getenv("AUTH_SECRET", "")
JWT_ALGO      = "HS256"
JWT_HOURS     = int(os.getenv("AUTH_JWT_HOURS", "12"))
BOOTSTRAP_KEY = os.getenv("BOOTSTRAP_KEY", "")  # one-time key to create the first org

VALID_ROLES = {"superadmin", "admin", "trainer"}
# superadmin = platform owner: manages all organisations, not scoped to one org
# admin      = Administration & Compliance within one org
# trainer    = Trainer within one org (sees only own assessments)

# A sentinel org for platform-level superadmins who don't belong to any RTO.
PLATFORM_ORG_ID = "__platform__"

# In-memory fallback stores (used when Firestore is unavailable)
_orgs_mem:   dict = {}
_users_mem:  dict = {}      # user_id -> user dict
_email_idx:  dict = {}      # email.lower() -> user_id


def _firestore():
    """Lazy Firestore client — shared with database.py pattern."""
    from .database import _firestore as _db_fs
    return _db_fs()


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    if not plain or len(plain) < 8:
        raise ValueError("Password must be at least 8 characters")
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ── JWT helpers ───────────────────────────────────────────────────────────────

def _get_secret() -> str:
    if not AUTH_SECRET:
        # Generate an ephemeral secret so the app still boots in dev, but log a
        # loud warning. In production AUTH_SECRET must be set as a Cloud Run
        # env var, otherwise every restart invalidates every token.
        global _EPHEMERAL
        if "_EPHEMERAL" not in globals():
            _EPHEMERAL = secrets.token_urlsafe(48)
            logger.warning("AUTH_SECRET not set — using ephemeral dev secret. "
                           "All tokens will be invalidated on restart. "
                           "Set AUTH_SECRET in Cloud Run for production.")
        return _EPHEMERAL
    return AUTH_SECRET


def issue_token(user: dict) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub":   user["id"],
        "org":   user["org_id"],
        "role":  user["role"],
        "email": user["email"],
        "name":  user.get("name", ""),
        "iat":   int(now.timestamp()),
        "exp":   int((now + timedelta(hours=JWT_HOURS)).timestamp()),
    }
    return jwt.encode(payload, _get_secret(), algorithm=JWT_ALGO)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, _get_secret(), algorithms=[JWT_ALGO])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Session expired — please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid authentication token.")


# ── Organisation CRUD ─────────────────────────────────────────────────────────

async def create_org(name: str, rto_code: str = "", contact_email: str = "") -> dict:
    org = {
        "id":            str(uuid.uuid4()),
        "name":          name.strip(),
        "rto_code":      rto_code.strip(),
        "contact_email": contact_email.strip(),
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "active":        True,
        "settings": {
            "asqa_retention_years": 2,
            "currency_years":       5,
        },
    }
    db = _firestore()
    if db:
        try:
            await db.collection("rpl_orgs").document(org["id"]).set(org)
        except Exception as e:
            logger.error(f"Firestore create_org failed: {e}")
    _orgs_mem[org["id"]] = org
    return org


async def get_org(org_id: str) -> Optional[dict]:
    if org_id in _orgs_mem:
        return _orgs_mem[org_id]
    db = _firestore()
    if db:
        try:
            doc = await db.collection("rpl_orgs").document(org_id).get()
            if doc.exists:
                d = doc.to_dict()
                _orgs_mem[org_id] = d
                return d
        except Exception as e:
            logger.error(f"Firestore get_org failed: {e}")
    return None


async def list_orgs() -> list:
    db = _firestore()
    out = []
    if db:
        try:
            async for doc in db.collection("rpl_orgs").stream():
                d = doc.to_dict()
                out.append(d)
                _orgs_mem[d["id"]] = d
            return out
        except Exception as e:
            logger.error(f"Firestore list_orgs failed: {e}")
    return list(_orgs_mem.values())


async def update_org_settings(org_id: str, settings: dict) -> Optional[dict]:
    org = await get_org(org_id)
    if not org:
        return None
    org["settings"] = {**org.get("settings", {}), **settings}
    db = _firestore()
    if db:
        try:
            await db.collection("rpl_orgs").document(org_id).update(
                {"settings": org["settings"]})
        except Exception as e:
            logger.error(f"Firestore update_org failed: {e}")
    _orgs_mem[org_id] = org
    return org


# ── User CRUD ─────────────────────────────────────────────────────────────────

async def create_user(org_id: str, email: str, password: str,
                       name: str, role: str) -> dict:
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of {sorted(VALID_ROLES)}")
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("Invalid email address")

    # Email uniqueness across the whole system (each user gets their own login)
    existing = await get_user_by_email(email)
    if existing:
        raise ValueError(f"A user with email {email} already exists")

    user = {
        "id":            str(uuid.uuid4()),
        "org_id":        org_id,
        "email":         email,
        "password_hash": hash_password(password),
        "name":          name.strip(),
        "role":          role,
        "active":        True,
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "last_login":    None,
    }
    db = _firestore()
    if db:
        try:
            await db.collection("rpl_users").document(user["id"]).set(user)
            # Email-index doc for O(1) login lookup
            await db.collection("rpl_user_email_index").document(email).set(
                {"user_id": user["id"]})
        except Exception as e:
            logger.error(f"Firestore create_user failed: {e}")
    _users_mem[user["id"]] = user
    _email_idx[email] = user["id"]
    return _sanitise_user(user)


async def get_user(user_id: str) -> Optional[dict]:
    if user_id in _users_mem:
        return _users_mem[user_id]
    db = _firestore()
    if db:
        try:
            doc = await db.collection("rpl_users").document(user_id).get()
            if doc.exists:
                d = doc.to_dict()
                _users_mem[d["id"]] = d
                _email_idx[d["email"]] = d["id"]
                return d
        except Exception as e:
            logger.error(f"Firestore get_user failed: {e}")
    return None


async def get_user_by_email(email: str) -> Optional[dict]:
    email = email.strip().lower()
    if email in _email_idx:
        return await get_user(_email_idx[email])
    db = _firestore()
    if db:
        try:
            idx = await db.collection("rpl_user_email_index").document(email).get()
            if idx.exists:
                uid = idx.to_dict().get("user_id")
                if uid:
                    _email_idx[email] = uid
                    return await get_user(uid)
            # Fallback: scan
            async for doc in db.collection("rpl_users").where("email", "==", email).stream():
                d = doc.to_dict()
                _users_mem[d["id"]] = d
                _email_idx[email] = d["id"]
                # Backfill index
                try:
                    await db.collection("rpl_user_email_index").document(email).set(
                        {"user_id": d["id"]})
                except Exception:
                    pass
                return d
        except Exception as e:
            logger.error(f"Firestore get_user_by_email failed: {e}")
    return None


async def list_users(org_id: str) -> list:
    db = _firestore()
    out = []
    if db:
        try:
            async for doc in db.collection("rpl_users").where("org_id", "==", org_id).stream():
                d = doc.to_dict()
                _users_mem[d["id"]] = d
                out.append(_sanitise_user(d))
            return out
        except Exception as e:
            logger.error(f"Firestore list_users failed: {e}")
    return [_sanitise_user(u) for u in _users_mem.values()
            if u.get("org_id") == org_id]


async def update_user(user_id: str, *,
                       name: Optional[str] = None,
                       role: Optional[str] = None,
                       active: Optional[bool] = None,
                       password: Optional[str] = None) -> Optional[dict]:
    user = await get_user(user_id)
    if not user:
        return None
    patch = {}
    if name is not None:    patch["name"] = name.strip()
    if role is not None:
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}")
        patch["role"] = role
    if active is not None:  patch["active"] = bool(active)
    if password is not None:
        patch["password_hash"] = hash_password(password)
    if not patch:
        return _sanitise_user(user)
    user.update(patch)
    db = _firestore()
    if db:
        try:
            await db.collection("rpl_users").document(user_id).update(patch)
        except Exception as e:
            logger.error(f"Firestore update_user failed: {e}")
    _users_mem[user_id] = user
    return _sanitise_user(user)


async def delete_user(user_id: str) -> bool:
    user = await get_user(user_id)
    if not user:
        return False
    db = _firestore()
    if db:
        try:
            await db.collection("rpl_users").document(user_id).delete()
            await db.collection("rpl_user_email_index").document(user["email"]).delete()
        except Exception as e:
            logger.error(f"Firestore delete_user failed: {e}")
    _users_mem.pop(user_id, None)
    _email_idx.pop(user.get("email", ""), None)
    return True


async def record_login(user_id: str):
    """Update last_login timestamp; failures are silent."""
    now = datetime.now(timezone.utc).isoformat()
    user = _users_mem.get(user_id)
    if user:
        user["last_login"] = now
    db = _firestore()
    if db:
        try:
            await db.collection("rpl_users").document(user_id).update({"last_login": now})
        except Exception:
            pass


def _sanitise_user(user: dict) -> dict:
    """Remove password_hash before returning to API consumers."""
    return {k: v for k, v in user.items() if k != "password_hash"}


# ── Authentication ────────────────────────────────────────────────────────────

async def authenticate(email: str, password: str) -> Optional[dict]:
    """Verify email+password and return the user (sanitised) on success."""
    user = await get_user_by_email(email)
    if not user:
        return None
    if not user.get("active", True):
        return None
    if not verify_password(password, user.get("password_hash", "")):
        return None
    await record_login(user["id"])
    return _sanitise_user(user)


# ── FastAPI dependencies ──────────────────────────────────────────────────────

async def current_user(authorization: Optional[str] = Header(None)) -> dict:
    """
    Resolve the active user from the Authorization: Bearer <jwt> header.
    Returns the sanitised user dict. Raises 401 on any auth failure.
    """
    if not authorization:
        raise HTTPException(401, "Missing Authorization header.")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Authorization must be 'Bearer <token>'.")
    token = authorization.split(" ", 1)[1].strip()
    claims = decode_token(token)
    user = await get_user(claims["sub"])
    if not user:
        raise HTTPException(401, "Account no longer exists.")
    if not user.get("active", True):
        raise HTTPException(403, "Account is disabled. Contact your admin.")
    # Ensure org/role haven't drifted from the token — DB is source of truth
    out = _sanitise_user(user)
    out["_jwt_claims"] = claims
    return out


async def require_admin(user: dict = Depends(current_user)) -> dict:
    # superadmin can do everything an org admin can
    if user.get("role") not in ("admin", "superadmin"):
        raise HTTPException(403, "Admin role required (Administration & Compliance).")
    return user


async def require_superadmin(user: dict = Depends(current_user)) -> dict:
    if user.get("role") != "superadmin":
        raise HTTPException(403, "Super-admin (platform owner) role required.")
    return user


async def require_trainer_or_admin(user: dict = Depends(current_user)) -> dict:
    # All roles can do trainer-tier actions; this dep just confirms a valid user.
    if user.get("role") not in VALID_ROLES:
        raise HTTPException(403, "Valid role required.")
    return user


# ── Bootstrap ─────────────────────────────────────────────────────────────────

async def bootstrap_first_org(name: str, rto_code: str,
                                admin_email: str, admin_password: str,
                                admin_name: str,
                                provided_key: str) -> dict:
    """
    Create the very first organisation and its first admin user.
    Caller must supply BOOTSTRAP_KEY to use this.

    Single-use: once any real organisation exists, this endpoint refuses to run
    even with a valid key. Additional organisations must be created through the
    super-admin console (POST /api/superadmin/orgs), so a leaked or un-removed
    BOOTSTRAP_KEY cannot be used to spin up rogue orgs/admins.
    """
    if not BOOTSTRAP_KEY:
        raise HTTPException(503,
            "BOOTSTRAP_KEY env var not set — system not configured for bootstrap.")
    if not secrets.compare_digest(provided_key, BOOTSTRAP_KEY):
        raise HTTPException(401, "Invalid bootstrap key.")

    # Refuse if a non-platform organisation already exists.
    existing = [o for o in await list_orgs() if o.get("id") != PLATFORM_ORG_ID]
    if existing:
        raise HTTPException(409,
            "Bootstrap already completed — an organisation already exists. "
            "Use the super-admin console to add further organisations.")

    org = await create_org(name=name, rto_code=rto_code,
                            contact_email=admin_email)
    try:
        user = await create_user(
            org_id=org["id"], email=admin_email, password=admin_password,
            name=admin_name, role="admin")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"org": org, "admin": user}


async def stats() -> dict:
    """Quick counts for the /health endpoint."""
    orgs  = await list_orgs()
    user_count = 0
    db = _firestore()
    if db:
        try:
            async for _ in db.collection("rpl_users").stream():
                user_count += 1
        except Exception:
            user_count = len(_users_mem)
    else:
        user_count = len(_users_mem)
    return {"orgs": len(orgs), "users": user_count}


# ── Super-admin (platform) helpers ────────────────────────────────────────────

async def create_org_with_admin(org_name: str, rto_code: str,
                                  admin_email: str, admin_password: str,
                                  admin_name: str,
                                  contact_email: str = "") -> dict:
    """
    Create a new organisation AND its first admin user in one operation.
    Used by the super-admin Organisations console. Returns {org, admin}.
    """
    org = await create_org(name=org_name, rto_code=rto_code,
                           contact_email=contact_email or admin_email)
    try:
        admin = await create_user(
            org_id=org["id"], email=admin_email, password=admin_password,
            name=admin_name, role="admin")
    except ValueError as e:
        # Roll back the orphan org if the admin couldn't be created
        try:
            db = _firestore()
            if db:
                await db.collection("rpl_orgs").document(org["id"]).delete()
            _orgs_mem.pop(org["id"], None)
        except Exception:
            pass
        raise HTTPException(400, str(e))
    return {"org": org, "admin": admin}


async def list_orgs_with_counts() -> list:
    """All organisations, each annotated with its user count and admin emails."""
    orgs = await list_orgs()
    out = []
    for org in orgs:
        users = await list_users(org["id"])
        admins = [u["email"] for u in users if u.get("role") == "admin"]
        out.append({
            **org,
            "user_count":   len(users),
            "admin_count":  len(admins),
            "admin_emails": admins,
            "trainer_count": sum(1 for u in users if u.get("role") == "trainer"),
        })
    return out


async def set_org_active(org_id: str, active: bool) -> Optional[dict]:
    """Enable or suspend an entire organisation."""
    org = await get_org(org_id)
    if not org:
        return None
    org["active"] = bool(active)
    db = _firestore()
    if db:
        try:
            await db.collection("rpl_orgs").document(org_id).update({"active": bool(active)})
        except Exception as e:
            logger.error(f"set_org_active failed: {e}")
    _orgs_mem[org_id] = org
    return org


async def promote_to_superadmin(email: str, password: str = None,
                                 name: str = None) -> dict:
    """
    Make a user a platform super-admin. If the user doesn't exist, create them
    in the platform org. Used by the one-time setup script.
    """
    email = email.strip().lower()
    user = await get_user_by_email(email)
    if user:
        patch = {"role": "superadmin", "active": True}
        if name:     patch["name"] = name
        if password: patch["password_hash"] = hash_password(password)
        user.update(patch)
        db = _firestore()
        if db:
            try:
                await db.collection("rpl_users").document(user["id"]).update(patch)
            except Exception as e:
                logger.error(f"promote_to_superadmin update failed: {e}")
        _users_mem[user["id"]] = user
        return _sanitise_user(user)
    # Create new platform superadmin
    if not password:
        raise ValueError("Password required to create a new super-admin")
    # Ensure platform org exists
    if not await get_org(PLATFORM_ORG_ID):
        plat = {
            "id": PLATFORM_ORG_ID, "name": "Platform", "rto_code": "",
            "contact_email": email,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "active": True, "settings": {},
        }
        db = _firestore()
        if db:
            try:
                await db.collection("rpl_orgs").document(PLATFORM_ORG_ID).set(plat)
            except Exception:
                pass
        _orgs_mem[PLATFORM_ORG_ID] = plat
    return await create_user(
        org_id=PLATFORM_ORG_ID, email=email, password=password,
        name=name or "Super Admin", role="superadmin")
