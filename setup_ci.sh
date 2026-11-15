#!/usr/bin/env bash
# Idempotent setup script that provisions the GCP Service Account and IAM roles
# for GitHub Actions CI/CD, and syncs secrets (GCP_SA_KEY, GEMINI_API_KEY,
# HEAR_AUTH_TOKEN) to the GitHub repository via the `gh` CLI.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

if [[ -z "${GEMINI_API_KEY:-}" || -z "${HEAR_AUTH_TOKEN:-}" ]]; then
  echo "✗ Error: Both GEMINI_API_KEY and HEAR_AUTH_TOKEN must be set in .env" >&2
  exit 1
fi

echo "▸ Ensuring IAM API is enabled on project '$PROJECT'..."
gcloud services enable iam.googleapis.com cloudresourcemanager.googleapis.com \
  --project "$PROJECT" \
  --quiet

# 2. Idempotently create the GitHub Actions deployer service account
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

# 3. Grant the exact IAM roles required to run ./deploy.sh from GitHub Actions
ROLES=(
  "roles/run.admin"
  "roles/cloudbuild.builds.editor"
  "roles/storage.admin"
  "roles/artifactregistry.admin"
  "roles/iam.serviceAccountUser"
  "roles/serviceusage.serviceUsageAdmin"
  "roles/resourcemanager.projectIamAdmin"
)

echo "▸ Binding IAM roles to '$SA_EMAIL'..."
for ROLE in "${ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$ROLE" \
    --condition=None \
    --quiet >/dev/null
done

# 4. Create a temporary service account key, upload to GitHub Secrets, and delete local key
TMP_KEY_FILE="$(mktemp -t hear-gcp-sa-key.XXXXXX.json)"
trap 'rm -f "$TMP_KEY_FILE"' EXIT

echo "▸ Generating service account key and syncing secrets to GitHub ($GH_REPO)..."
gcloud iam service-accounts keys create "$TMP_KEY_FILE" \
  --iam-account="$SA_EMAIL" \
  --project "$PROJECT" \
  --quiet

gh secret set GCP_SA_KEY --repo "$GH_REPO" < "$TMP_KEY_FILE"
gh secret set GEMINI_API_KEY --repo "$GH_REPO" --body "$GEMINI_API_KEY"
gh secret set HEAR_AUTH_TOKEN --repo "$GH_REPO" --body "$HEAR_AUTH_TOKEN"

echo "✓ CI/CD setup complete!"
echo "  - GCP Service Account: $SA_EMAIL"
echo "  - GitHub Secrets set on $GH_REPO: GCP_SA_KEY, GEMINI_API_KEY, HEAR_AUTH_TOKEN"
