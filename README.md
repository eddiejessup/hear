# Hear — Free-Roam Historical Walking Companion

A personal, location-aware historical audio companion powered by **Gemini** (`gemini-3.8-flash` / `gemini-4-argon` with Google Search grounding + `gemini-3.8-flash-tts`).

## Project Structure
```text
hear/
├── config.yaml              # Hot-reloaded lenses, system prompt, model/voice, and trigger rules
├── Dockerfile               # Cloud Run / container image definition
├── pyproject.toml           # Python 3.15 project config (uv, ruff, ty)
├── deploy/
│   ├── check.sh             # Static analysis runner (ruff format, ruff check, ty check)
│   ├── deploy.sh            # Cloud Run deployment script
│   └── setup_ci.sh          # GCP Service Account & GitHub Actions Secrets provisioning
└── src/
    └── hear/
        ├── __init__.py
        ├── app.py           # FastAPI application, endpoints, and CLI entry point
        ├── auth.py          # Fail-closed token authentication helper
        ├── config.py        # Hot-reloading config.yaml manager
        ├── debug.py         # Build, git commit, Cloud Run revision, and runtime telemetry
        ├── gemini.py        # Gemini generateContent (with Google Search) & TTS client
        ├── geo.py           # Haversine distance, compass bearings, Nominatim & Overpass lookups
        ├── paths.py         # Project paths & directory initialization
        ├── session.py       # WalkSession state machine, lens sampling & trigger heuristics
        ├── models/
        │   ├── __init__.py
        │   ├── api.py       # HTTP request & response models
        │   ├── common.py    # Shared HearModel base configuration
        │   ├── config.py    # Configuration models (AppConfig, GeminiConfig, TriggersConfig, LensConfig)
        │   ├── gemini.py    # Gemini API payload & grounding models
        │   ├── geo.py       # Reverse geocoding context model
        │   └── session.py   # Walk session, messages, trigger status & DebugInfo models
        └── static/
            └── index.html   # Mobile web UI
```

## Quick Start (Local + Tailscale)

### 1. Sync environment (Python 3.15)
```bash
uv sync
```

### 2. Configure `.env`
For local or private Tailscale use, you can either set a token (`HEAR_AUTH_TOKEN`) or explicitly allow unauthenticated local access (`HEAR_ALLOW_UNAUTHENTICATED="1"`):
```bash
cat << 'EOF' > .env
GEMINI_API_KEY="your-api-key-here"
HEAR_ALLOW_UNAUTHENTICATED="1"
EOF
```

### 3. Run the server
```bash
uv run hear
```

### 4. Static Analysis (`ruff` & `ty`)
```bash
./deploy/check.sh
```

---

## Deploying to Google Cloud Run

The server is **fail-closed** by default: when `HEAR_ALLOW_UNAUTHENTICATED` is not set, all `/api/*` endpoints require `HEAR_AUTH_TOKEN` (via `x-hear-token` header or `SameSite=Strict` cookie).

```bash
# Deploy directly from your machine:
./deploy/deploy.sh

# Or provision GitHub Actions CI/CD secrets (one-time):
./deploy/setup_ci.sh
```
