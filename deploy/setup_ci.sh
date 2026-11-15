#!/usr/bin/env bash
# Idempotent project & CI/CD bootstrap script:
# 1. Enables all required GCP APIs (including serviceusage.googleapis.com)
# 2. Configures Cloud Build permissions on the default Compute Engine service account
# 3. Provisions the GitHub Actions deployer service account & IAM roles
# 4. Syncs secrets (GCP_SA_KEY, GEMINI_API_KEY, HEAR_AUTH_TOKEN) to GitHub Actions
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PATH="/opt/homebrew/bin:/opt/homebrew/share/google-cloud-sdk/bin:$PATH"

PROJECT="${GCP_PROJECT:-gen-lang-client-0743073374}"
GH_REPO="${GH_REPO:-eddiejessup/hear}"
SA_NAME="github-deployer"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

# 1. Load .env for GEMINI_API_KEY and HEAR_AUTH_TOKEN
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "✗ Error: GEMINI_API_KEY must be set in .env" >&2
  exit 1
fi

if [[ -z "${HEAR_AUTH_TOKEN:-}" ]]; then
  HEAR_AUTH_TOKEN="$(openssl rand -hex 24)"
  echo "HEAR_AUTH_TOKEN=\"$HEAR_AUTH_TOKEN\"" >> .env
  echo "▸ Generated new HEAR_AUTH_TOKEN and saved it to .env"
fi

echo "▸ Enabling required GCP APIs on project '$PROJECT'..."
gcloud services enable \
  --project "$PROJECT" \
  --quiet \
  serviceusage.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com

# 2. Configure Cloud Build roles on the default Compute Engine service account
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
COMPUTE_SA="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "▸ Ensuring Cloud Build IAM roles on ${COMPUTE_SA}..."
for ROLE in "roles/cloudbuild.builds.builder" "roles/serviceusage.serviceUsageConsumer"; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="$COMPUTE_SA" \
    --role="$ROLE" \
    --condition=None \
    --quiet >/dev/null
done

# 3. Idempotently create the GitHub Actions deployer service account
if gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1; then
  echo "▸ Service account '$SA_EMAIL' already exists."
else
  echo "▸ Creating service account '$SA_EMAIL'..."
  gcloud iam service-accounts create "$SA_NAME" \
    --project "$PROJECT" \
    --display-name "GitHub Actions Cloud Run Deployer (hear)" \
    --quiet
  echo "▸ Waiting 10s for GCP IAM service account propagation..."
  sleep 10
fi

# 4. Grant the exact IAM roles required to run ./deploy/deploy.sh from GitHub Actions
ROLES=(
  "roles/run.admin"
  "roles/cloudbuild.builds.editor"
  "roles/storage.admin"
  "roles/artifactregistry.admin"
  "roles/iam.serviceAccountUser"
  "roles/serviceusage.serviceUsageConsumer"
  "roles/logging.viewer"
)

echo "▸ Binding IAM roles to '$SA_EMAIL'..."
for ROLE in "${ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$ROLE" \
    --condition=None \
    --quiet >/dev/null
done

# 5. If a new key isn't explicitly requested and GCP_SA_KEY already exists on GitHub, skip key creation
if [[ "${ROTATE_SA_KEY:-0}" == "1" ]] || ! gh secret list --repo "$GH_REPO" | grep -q "^GCP_SA_KEY"; then
  TMP_KEY_FILE="$(mktemp -t hear-gcp-sa-key.XXXXXX.json)"
  trap 'rm -f "$TMP_KEY_FILE"' EXIT

  echo "▸ Generating service account key and uploading GCP_SA_KEY to GitHub ($GH_REPO)..."
  gcloud iam service-accounts keys create "$TMP_KEY_FILE" \
    --iam-account="$SA_EMAIL" \
    --project "$PROJECT" \
    --quiet

  gh secret set GCP_SA_KEY --repo "$GH_REPO" < "$TMP_KEY_FILE"
else
  echo "▸ GCP_SA_KEY already set on $GH_REPO (set ROTATE_SA_KEY=1 to rotate)."
fi

gh secret set GEMINI_API_KEY --repo "$GH_REPO" --body "$GEMINI_API_KEY"
gh secret set HEAR_AUTH_TOKEN --repo "$GH_REPO" --body "$HEAR_AUTH_TOKEN"

echo "✓ GCP project & CI/CD setup complete!"
