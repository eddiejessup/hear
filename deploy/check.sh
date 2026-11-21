#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"

if [[ -x ".venv/bin/ruff" && -x ".venv/bin/ty" ]]; then
  RUFF=".venv/bin/ruff"
  TY=".venv/bin/ty"
else
  RUFF="uv run ruff"
  TY="uv run ty"
fi

BIOME="$(command -v biome || true)"

if [[ "${1:-}" == "--fix" ]]; then
  echo "▸ Running ruff format..."
  $RUFF format .
  echo "▸ Running ruff check --fix..."
  $RUFF check --fix .
  if [[ -n "$BIOME" ]]; then
    echo "▸ Running biome check --write (JS/CSS)..."
    "$BIOME" check --write src/hear/static
  fi
else
  echo "▸ Checking formatting (ruff format --check)..."
  $RUFF format --check .
  echo "▸ Running linter (ruff check)..."
  $RUFF check .
  if [[ -n "$BIOME" ]]; then
    echo "▸ Running frontend checks (biome check --error-on-warnings)..."
    "$BIOME" check --error-on-warnings src/hear/static
  elif [[ "${CI:-}" == "true" ]]; then
    echo "✗ Error: biome is required in CI but not found on PATH" >&2
    exit 1
  else
    echo "⚠ biome not found (install via 'brew install biome' to lint/format JS & CSS)"
  fi
fi

echo "▸ Running type checker (ty check)..."
$TY check

if [[ -x ".venv/bin/python" ]] && .venv/bin/python -V >/dev/null 2>&1; then
  echo "▸ Running unit tests..."
  .venv/bin/python -m unittest discover -s tests -v
elif command -v uv >/dev/null 2>&1 && uv run python -V >/dev/null 2>&1; then
  echo "▸ Running unit tests..."
  uv run python -m unittest discover -s tests -v
fi

echo "✓ All static analysis checks passed."
