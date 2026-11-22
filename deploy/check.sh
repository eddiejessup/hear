#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"

if [[ -x ".venv/bin/ruff" && -x ".venv/bin/ty" && -x ".venv/bin/deptry" ]]; then
  RUFF=(".venv/bin/ruff")
  TY=(".venv/bin/ty")
  DEPTRY=(".venv/bin/deptry")
else
  RUFF=(uv run ruff)
  TY=(uv run ty)
  DEPTRY=(uv run deptry)
fi

if [[ -x ".venv/bin/shellcheck" ]]; then
  SHELLCHECK=".venv/bin/shellcheck"
else
  SHELLCHECK="$(command -v shellcheck || true)"
fi

BIOME="$(command -v biome || true)"
ACTIONLINT="$(command -v actionlint || true)"
HADOLINT="$(command -v hadolint || true)"
GITLEAKS="$(command -v gitleaks || true)"

require_or_warn() {
  local name="$1"
  local hint="$2"
  if [[ "${CI:-}" == "true" ]]; then
    echo "✗ Error: $name is required in CI but not found on PATH" >&2
    exit 1
  fi
  echo "⚠ $name not found ($hint)"
}

if [[ "${1:-}" == "--fix" ]]; then
  echo "▸ Running ruff format..."
  "${RUFF[@]}" format .
  echo "▸ Running ruff check --fix..."
  "${RUFF[@]}" check --fix .
  if [[ -n "$BIOME" ]]; then
    echo "▸ Running biome check --write (JS/CSS)..."
    "$BIOME" check --write src/hear/static
  fi
else
  echo "▸ Checking formatting (ruff format --check)..."
  "${RUFF[@]}" format --check .
  echo "▸ Running linter (ruff check)..."
  "${RUFF[@]}" check .
  if [[ -n "$BIOME" ]]; then
    echo "▸ Running frontend checks (biome check --error-on-warnings)..."
    "$BIOME" check --error-on-warnings src/hear/static
  else
    require_or_warn "biome" "install via 'brew install biome' to lint/format JS & CSS"
  fi
fi

if [[ -n "$SHELLCHECK" ]]; then
  echo "▸ Running shell script checks (shellcheck)..."
  "$SHELLCHECK" -x deploy/*.sh .githooks/pre-commit
else
  require_or_warn "shellcheck" "run 'uv sync' or 'brew install shellcheck' to lint shell scripts"
fi

if [[ -n "$ACTIONLINT" ]]; then
  echo "▸ Running GitHub Actions checks (actionlint)..."
  "$ACTIONLINT"
else
  require_or_warn "actionlint" "install via 'brew install actionlint' to lint GitHub Actions"
fi

if [[ -n "$HADOLINT" ]]; then
  echo "▸ Running Dockerfile checks (hadolint)..."
  "$HADOLINT" Dockerfile
else
  require_or_warn "hadolint" "install via 'brew install hadolint' to lint Dockerfile"
fi

if [[ -n "$GITLEAKS" ]]; then
  echo "▸ Running secret scan (gitleaks)..."
  "$GITLEAKS" git --no-banner --redact
  "$GITLEAKS" git --staged --no-banner --redact
else
  require_or_warn "gitleaks" "install via 'brew install gitleaks' to scan for secrets"
fi

echo "▸ Running dependency checks (deptry)..."
"${DEPTRY[@]}" .

echo "▸ Running type checker (ty check)..."
"${TY[@]}" check

if [[ -x ".venv/bin/pytest" ]] && .venv/bin/pytest --version >/dev/null 2>&1; then
  echo "▸ Running unit tests..."
  .venv/bin/pytest -v
elif command -v uv >/dev/null 2>&1 && uv run pytest --version >/dev/null 2>&1; then
  echo "▸ Running unit tests..."
  uv run pytest -v
fi

echo "✓ All static analysis checks passed."
