#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "$ROOT_DIR/deploy/_common.sh"

SERVICE_NAME="${SERVICE_NAME:-hear}"

if [[ "${1:-}" != "--skip-checks" ]]; then
  echo "▸ Running pre-deploy static analysis checks..."
  ./deploy/check.sh
fi

ensure_deploy_secrets

GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT_TIME="$(git show -s --format=%cI HEAD 2>/dev/null || echo 'unknown')"
BUILD_TIME="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

echo "▸ Deploying '$SERVICE_NAME' to Cloud Run (project: $PROJECT, region: $REGION, bucket: gs://$LITESTREAM_BUCKET, commit: $GIT_COMMIT)..."
gcloud run deploy "$SERVICE_NAME" \
  --project "$PROJECT" \
  --quiet \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --execution-environment gen2 \
  --no-cpu-throttling \
  --max-instances 1 \
  --memory 512Mi \
  --cpu 1 \
  --timeout 120 \
  --set-env-vars "GEMINI_API_KEY=${GEMINI_API_KEY},HEAR_AUTH_TOKEN=${HEAR_AUTH_TOKEN},HEAR_LITESTREAM_BUCKET=${LITESTREAM_BUCKET},HEAR_GIT_COMMIT=${GIT_COMMIT},HEAR_GIT_COMMIT_TIME=${GIT_COMMIT_TIME},HEAR_BUILD_TIME=${BUILD_TIME}"

echo ""
echo "✓ Deployment complete!"
