#!/usr/bin/env python3
"""
make_superadmin.py — one-time setup to create or promote a platform super-admin.

Run in Cloud Shell from the project root:

    pip install --quiet google-cloud-firestore bcrypt
    python3 make_superadmin.py mstaal@abctraining.edu.au "ChooseAStrongPassword" "Michael Staal"

If the email already exists, it is promoted to super-admin (password updated
only if you pass one). If it doesn't exist, a new platform super-admin is created.

The GCP project is read from the GOOGLE_CLOUD_PROJECT env var, or falls back to
the hard-coded default below — edit it if yours differs.
"""
import sys, os
from datetime import datetime, timezone

import bcrypt
from google.cloud import firestore

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")  # inferred from ADC if unset
PLATFORM_ORG_ID = "__platform__"


def hash_pw(plain: str) -> str:
    if len(plain) < 8:
        sys.exit("ERROR: password must be at least 8 characters")
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 make_superadmin.py <email> [password] [name]")
    email = sys.argv[1].strip().lower()
    password = sys.argv[2] if len(sys.argv) > 2 else None
    name = sys.argv[3] if len(sys.argv) > 3 else "Super Admin"

    db = firestore.Client(project=PROJECT)

    # Find existing user by email
    existing = None
    for doc in db.collection("rpl_users").where("email", "==", email).stream():
        existing = doc
        break

    if existing:
        patch = {"role": "superadmin", "active": True, "name": name}
        if password:
            patch["password_hash"] = hash_pw(password)
        db.collection("rpl_users").document(existing.id).update(patch)
        print(f"✓ Promoted existing user {email} to super-admin.")
        if password:
            print(f"  Password updated.")
        else:
            print(f"  Password unchanged — log in with their existing password.")
        return

    # Create new platform super-admin
    if not password:
        sys.exit("ERROR: user does not exist yet — supply a password to create them:\n"
                 f"  python3 make_superadmin.py {email} 'AStrongPassword' '{name}'")

    # Ensure the platform org exists
    plat_ref = db.collection("rpl_orgs").document(PLATFORM_ORG_ID)
    if not plat_ref.get().exists:
        plat_ref.set({
            "id": PLATFORM_ORG_ID, "name": "Platform", "rto_code": "",
            "contact_email": email,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "active": True, "settings": {},
        })

    import uuid
    uid = str(uuid.uuid4())
    user = {
        "id": uid, "org_id": PLATFORM_ORG_ID, "email": email,
        "password_hash": hash_pw(password), "name": name,
        "role": "superadmin", "active": True,
        "created_at": datetime.now(timezone.utc).isoformat(), "last_login": None,
    }
    db.collection("rpl_users").document(uid).set(user)
    db.collection("rpl_user_email_index").document(email).set({"user_id": uid})
    print(f"✓ Created new platform super-admin: {email}")


if __name__ == "__main__":
    main()
