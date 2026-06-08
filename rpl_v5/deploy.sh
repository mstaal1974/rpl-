#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# RPL System v5 — Cloud Run deploy helper
#
# Run from Google Cloud Shell or any machine with gcloud installed and
# authenticated to your project.
#
# Usage:
#   PROJECT=my-gcp-project  REGION=australia-southeast1  ./deploy.sh
#
# All variables below have sensible defaults; override any by exporting them
# before running.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-australia-southeast1}"
VERTEX_REGION="${VERTEX_REGION:-us-east5}"
SERVICE="${SERVICE:-rpl-portal}"

if [[ -z "$PROJECT" ]]; then
  echo "ERROR: PROJECT is not set. Run: gcloud config set project YOUR-PROJECT"
  exit 1
fi

echo "──────────────────────────────────────────────"
echo "  Project:        $PROJECT"
echo "  Region:         $REGION"
echo "  Vertex region:  $VERTEX_REGION"
echo "  Service:        $SERVICE"
echo "──────────────────────────────────────────────"

# 1) Enable required APIs (idempotent)
echo "▸ enabling APIs..."
gcloud services enable \
  run.googleapis.com \
  aiplatform.googleapis.com \
  firestore.googleapis.com \
  bigquery.googleapis.com \
  texttospeech.googleapis.com \
  cloudbuild.googleapis.com \
  --project "$PROJECT" --quiet

# 2) Make sure Firestore is in Native mode for this project (one-time)
if ! gcloud firestore databases describe --database='(default)' --project "$PROJECT" --quiet >/dev/null 2>&1; then
  echo "▸ creating Firestore database in $REGION..."
  gcloud firestore databases create --location="$REGION" --project "$PROJECT" --quiet
fi

# 3) Generate strong secrets if not already set
AUTH_SECRET="${AUTH_SECRET:-$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')}"
BOOTSTRAP_KEY="${BOOTSTRAP_KEY:-$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')}"

echo ""
echo "▸ deploying to Cloud Run..."

# Comma-separated env vars — values must NOT contain commas
ENV_VARS="\
GOOGLE_CLOUD_PROJECT=$PROJECT,\
VERTEX_REGION=$VERTEX_REGION,\
AUTH_SECRET=$AUTH_SECRET,\
BOOTSTRAP_KEY=$BOOTSTRAP_KEY,\
AUTH_JWT_HOURS=12,\
ASQA_RETENTION_YEARS=2,\
FIRESTORE_DATABASE=(default),\
BQ_DATASET=rpl_audit,\
BQ_TABLE=assessment_records"

gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT" \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10 \
  --set-env-vars "$ENV_VARS"

# 4) Get the URL and set BASE_URL so invite links work
URL=$(gcloud run services describe "$SERVICE" \
  --region "$REGION" --project "$PROJECT" --format='value(status.url)')

echo "▸ setting BASE_URL=$URL..."
gcloud run services update "$SERVICE" \
  --region "$REGION" --project "$PROJECT" \
  --update-env-vars "BASE_URL=$URL" --quiet >/dev/null

# 5) Grant the service identity Vertex AI access
SA=$(gcloud run services describe "$SERVICE" \
  --region "$REGION" --project "$PROJECT" \
  --format='value(spec.template.spec.serviceAccountName)')
SA="${SA:-${PROJECT_NUMBER:-$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')}-compute@developer.gserviceaccount.com}"
echo "▸ granting Vertex AI user role to $SA..."
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member "serviceAccount:$SA" \
  --role roles/aiplatform.user --condition=None --quiet >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member "serviceAccount:$SA" \
  --role roles/datastore.user --condition=None --quiet >/dev/null

cat <<EOF

══════════════════════════════════════════════════════════════════
 ✓ Deployed.

   URL:           $URL
   AUTH_SECRET:   $AUTH_SECRET
   BOOTSTRAP_KEY: $BOOTSTRAP_KEY

   *** SAVE BOTH OF THE ABOVE SOMEWHERE SAFE ***

 Now create your first organisation and admin account:

   curl -X POST $URL/api/auth/bootstrap \\
     -H 'Content-Type: application/json' \\
     -d '{
       "bootstrap_key":  "$BOOTSTRAP_KEY",
       "org_name":       "Your RTO name",
       "rto_code":       "12345",
       "admin_email":    "you@your-domain.com.au",
       "admin_password": "a-strong-password",
       "admin_name":     "Your Name"
     }'

 Then open $URL and log in with that email + password.

 IMPORTANT: Once you've created your first admin, remove BOOTSTRAP_KEY
 from the Cloud Run environment so no one else can create rogue orgs:

   gcloud run services update $SERVICE \\
     --region $REGION --project $PROJECT \\
     --remove-env-vars BOOTSTRAP_KEY

══════════════════════════════════════════════════════════════════
EOF
