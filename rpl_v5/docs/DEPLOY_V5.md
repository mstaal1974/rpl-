# RPL System v5 — Multi-tenant Deployment Guide

## What changed from v4

| Area | v4 | v5 |
|---|---|---|
| Auth | One shared `TRAINER_PIN` for everyone | Per-user email + password login, JWT sessions |
| Tenancy | Hard-coded "ABC Training RTO #5800" | Each organisation isolated by `org_id` |
| Roles | Trainer-only | **Administration & Compliance** (admin) + **Trainers** (user) |
| Org admin UI | None | Built-in "Users" page for admins to manage staff |
| Audit | Hard-coded RTO label in BigQuery | Stamped with `org_id` |

## Data model

```
Organisation                    (one row per RTO)
  ├─ id, name, rto_code, settings
  └─ Users  (many)              (one row per real person)
        ├─ role = "admin"       → Administration & Compliance
        ├─ role = "trainer"     → Trainers
        └─ each user has their own email + password
```

- **Admins** can: manage users in their org, manage units, see every assessment in the org, change org settings, plus everything trainers can do.
- **Trainers** can: create their own assessments, invite candidates, review and mark their own submissions. They see only their own work.
- **Tenant isolation:** all DB queries are scoped by `org_id`. A user from org A cannot see, list, or fetch anything from org B — they get a 404 if they try.

## Deploy in 30 seconds

From Google Cloud Shell (or any shell with `gcloud` authenticated):

```bash
unzip rpl_v5.zip && cd rpl_v5
gcloud config set project YOUR-GCP-PROJECT
./deploy.sh
```

The script:
1. Enables the required APIs (Run, Vertex AI, Firestore, BigQuery, TTS, Cloud Build)
2. Creates the Firestore database in Native mode if needed
3. Generates a strong `AUTH_SECRET` and `BOOTSTRAP_KEY`
4. Builds and deploys to Cloud Run
5. Sets `BASE_URL` to the public URL so invite links work
6. Grants the service account `roles/aiplatform.user` and `roles/datastore.user`
7. Prints the curl command to create your first organisation

## First-time setup — create your organisation and admin

After `./deploy.sh` prints the URL and the bootstrap key, run the curl it printed. Example:

```bash
curl -X POST https://rpl-portal-xxxxx.australia-southeast1.run.app/api/auth/bootstrap \
  -H 'Content-Type: application/json' \
  -d '{
    "bootstrap_key":  "the-key-deploy-sh-printed",
    "org_name":       "Greenhills Training Group",
    "rto_code":       "12345",
    "admin_email":    "jane.smith@greenhills.edu.au",
    "admin_password": "use-a-strong-password-here",
    "admin_name":     "Jane Smith"
  }'
```

Then open the URL in your browser and log in with `jane.smith@greenhills.edu.au` and the password you chose.

### Lock down bootstrap

Once you've created your first admin, **remove the bootstrap key** so nobody else can use it:

```bash
gcloud run services update rpl-portal \
  --region australia-southeast1 \
  --remove-env-vars BOOTSTRAP_KEY
```

You can re-add it temporarily later if you ever need to onboard another organisation through the API.

## Day-to-day: how each role uses the portal

### Administration & Compliance (admin role)
1. Log in.
2. Go to **Users** in the sidebar — invite new trainers or other admins. Each person gets their own email + password.
3. Go to **Upload training packages** to load units of competency.
4. From the dashboard, see every assessment in the organisation. You can review and mark complete on behalf of any trainer.

### Trainers (trainer role)
1. Log in.
2. Click **New RPL assessment**, fill in candidate details, select units, send the invite.
3. Watch the dashboard — see *only your own* assessments. When a student submits, review and mark complete.

### Students (unchanged from v4)
Open the invite link, work through the steps, submit. No login needed — the invite token is their authentication.

## Adding more organisations

`/api/auth/bootstrap` is **single-use** — once your first organisation exists it
refuses to run again, even with a valid `BOOTSTRAP_KEY`. This prevents a leaked
or un-removed key from being used to create rogue orgs and admins.

To add further organisations, use the **super-admin console**:

- Promote a platform owner once with `make_superadmin.py` (see that script's header).
- Then create organisations (each with their first admin) via
  `POST /api/superadmin/orgs`, or from the **Organisations** page in the portal
  when logged in as the super-admin.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | yes | GCP project ID |
| `VERTEX_REGION` | yes | e.g. `us-east5` |
| `AUTH_SECRET` | yes | 48+ chars of randomness, used to sign JWTs |
| `BOOTSTRAP_KEY` | one-time | needed only to call `/api/auth/bootstrap`; remove after |
| `BASE_URL` | yes | Your Cloud Run URL — used in invite links |
| `AUTH_JWT_HOURS` | no | JWT lifetime in hours (default 12) |
| `ASQA_RETENTION_YEARS` | no | Default 2 |
| `SENDGRID_API_KEY` | no | Email notifications on invite/submit |
| `SENDGRID_FROM` | no | From address for emails |
| `DOCAI_DEFAULT_PROCESSOR_ID` | no | Document AI processor for OCR |

## AI analysis — what changed

The v4 code had a real bug: when a trainer or student triggered knowledge analysis, the candidate's role/employer/industry context was loaded from the assessment but **silently not forwarded** to the AI prompt builder, so all analysis ran without sector awareness. Fixed in v5 — both call sites now pass the candidate dict, and the prompt builder adds a `Candidate: Role / Employer / Sector` line before the rubric. Sector-appropriate analysis is now active.

The orchestrator multi-agent path (Evidence Intake → Knowledge → Element Mapping → Gap → Cross-unit → Synthesis) was already passing candidate through; that path was unaffected.

## Updating after code changes

```bash
cd rpl_v5
gcloud run deploy rpl-portal --source . --region australia-southeast1
```

Env vars persist across redeploys. Migration: v4 records (no `org_id`) continue to work for read-only access but won't appear in any org's list (they're invisible). If you need to migrate v4 data, write a one-shot Firestore script to stamp `org_id` and `trainer_user_id` on the legacy docs.

## Troubleshooting

- **"Missing Authorization header"** — your session has expired or you're not logged in. The login screen should appear automatically; if it doesn't, do a hard refresh (Ctrl+Shift+R).
- **"Invalid email or password"** — case-sensitive on password, case-insensitive on email.
- **`/health` shows `firestore: unconfigured`** — the service account doesn't have `roles/datastore.user`. `deploy.sh` grants it; if you deployed manually, run the IAM binding step.
- **AI analysis returns 500** — check Cloud Run logs. Most common cause is the service account missing `roles/aiplatform.user`.
- **"BOOTSTRAP_KEY env var not set"** when you try to bootstrap — you removed it after first use. Re-add it temporarily.
