#!/usr/bin/env bash
# Shared environment loading & credential checks for deploy/deploy.sh and deploy/setup_ci.sh.
# Intended to be sourced from a script that has already cd'd to ROOT_DIR.

export PATH="/opt/homebrew/bin:/opt/homebrew/share/google-cloud-sdk/bin:$PATH"

# Load .env if present (before applying defaults so .env overrides work)
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

PROJECT="${GCP_PROJECT:-gen-lang-client-0743073374}"
REGION="${GCP_REGION:-europe-west1}"
LITESTREAM_BUCKET="${HEAR_LITESTREAM_BUCKET:-${PROJECT}-hear-data}"

ensure_deploy_secrets() {
  if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "✗ Error: GEMINI_API_KEY is not set in environment or .env" >&2
    exit 1
  fi

  if [[ -z "${HEAR_AUTH_TOKEN:-}" ]]; then
    HEAR_AUTH_TOKEN="$(openssl rand -hex 24)"
    echo "HEAR_AUTH_TOKEN=\"$HEAR_AUTH_TOKEN\"" >> .env
    echo "▸ Generated new HEAR_AUTH_TOKEN and saved it to .env"
  fi
}
