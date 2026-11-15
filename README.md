# Hear — Free-Roam Historical Walking Companion

A personal, location-aware historical audio companion powered by **Gemini** (`gemini-3.8-flash` / `gemini-4-argon` with Google Search grounding + `gemini-3.8-flash-tts`).

## Project Structure
```text
hear/
├── config.yaml              # Hot-reloaded lenses, system prompt, model/voice, and trigger rules
├── Dockerfile               # Cloud Run / container image definition
├── pyproject.toml           # Python 3.15 project config (uv, ruff, ty)
└── src/
    └── hear/
        ├── __init__.py
        ├── app.py           # FastAPI application, endpoints, and CLI entry point
        ├── auth.py          # Fail-closed token authentication helper
        ├── config.py        # Hot-reloading config.yaml manager
        ├── gemini.py        # Gemini generateContent (with Google Search) & TTS client
        ├── geo.py           # Haversine distance, compass bearings, Nominatim & Overpass lookups
        ├── models.py        # Pydantic & TypedDict schemas
        ├── paths.py         # Project paths & directory initialization
        ├── session.py       # WalkSession state machine, lens sampling & trigger heuristics
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
./check.sh
```

---

## Deploying to Google Cloud Run

The server is **fail-closed** by default: when `HEAR_ALLOW_UNAUTHENTICATED` is not set, all `/api/*` endpoints require `HEAR_AUTH_TOKEN` (via `x-hear-token` header or `SameSite=Strict` cookie). Your browser prompts once on first visit and saves the token locally.

```bash
# 1. Generate a random access token
export HEAR_AUTH_TOKEN="$(openssl rand -hex 24)"
echo "Your access token: $HEAR_AUTH_TOKEN"

# 2. Deploy to Cloud Run (capped at max-instances=1 for stateful walk sessions & billing safety)
gcloud run deploy hear \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --max-instances 1 \
  --set-env-vars "GEMINI_API_KEY=your-gemini-key,HEAR_AUTH_TOKEN=$HEAR_AUTH_TOKEN"
```
