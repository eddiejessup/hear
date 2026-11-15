#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/bin/ruff" && -x ".venv/bin/ty" ]]; then
  RUFF=".venv/bin/ruff"
  TY=".venv/bin/ty"
else
  export PATH="$HOME/.local/bin:$PATH"
  RUFF="uv run ruff"
  TY="uv run ty"
fi

if [[ "${1:-}" == "--fix" ]]; then
  echo "▸ Running ruff format..."
  $RUFF format .
  echo "▸ Running ruff check --fix..."
  $RUFF check --fix .
else
  echo "▸ Checking formatting (ruff format --check)..."
  $RUFF format --check .
  echo "▸ Running linter (ruff check)..."
  $RUFF check .
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
