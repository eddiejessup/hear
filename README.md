# Hear — Free-Roam Historical Walking Companion

A personal, location-aware historical audio companion powered by **Gemini** (`gemini-4-hydrogen` / `gemini-4-argon` with Google Search grounding + `gemini-3.8-flash-tts`).

## Project Structure
```text
hear/
├── biome.json               # Frontend JS/CSS lint & format config
├── config.yaml              # Hot-reloaded lenses, system prompt, model/voice, and trigger rules
├── Dockerfile               # Cloud Run / container image definition
├── pyproject.toml           # Python 3.15 project config (uv, ruff, ty, pytest)
├── deploy/
│   ├── _common.sh           # Shared .env, GCP defaults & secret helpers for deploy scripts
│   ├── check.sh             # Verification runner (ruff, biome, ty, pytest)
│   ├── deploy.sh            # Cloud Run deployment script (Gen2, Litestream-enabled)
│   ├── entrypoint.sh        # Container startup wrapper (optional Litestream restore/replicate)
│   ├── litestream.yml       # Litestream GCS replication config for hear.db
│   └── setup_ci.sh          # GCP APIs, Litestream GCS bucket, SA & GitHub Secrets provisioning
├── src/
│   └── hear/
│       ├── __init__.py
│       ├── app.py           # FastAPI application, AppContainer DI, endpoints & CLI entry point
│       ├── auth.py          # Fail-closed token authentication & .env reader
│       ├── config.py        # Hot-reloading config.yaml & SQLite override manager
│       ├── db.py            # SQLite store (WAL, FTS5) for sessions, messages, events & config
│       ├── debug.py         # Build, git commit, Cloud Run revision & runtime telemetry
│       ├── gemini.py        # Gemini generateContent (with Google Search) & TTS client
│       ├── geo.py           # Haversine distance, compass bearings & Nominatim reverse geocoding
│       ├── http_client.py   # Shared lazy httpx.AsyncClient ownership wrapper
│       ├── logging_utils.py # Structured JSON logging & EventLogger with ContextVar correlation
│       ├── paths.py         # Project paths & directory initialization
│       ├── session.py       # WalkSession state machine, lens sampling & trigger heuristics
│       ├── models/
│       │   ├── __init__.py
│       │   ├── api.py       # HTTP request & response models
│       │   ├── common.py    # Shared HearModel base configuration
│       │   ├── config.py    # Configuration models (AppConfig, GeminiConfig, TriggersConfig, LensConfig)
│       │   ├── events.py    # Event source, level & backend event type enums
│       │   ├── gemini.py    # Gemini API payload, usage & grounding models
│       │   ├── geo.py       # Reverse geocoding context model
│       │   └── session.py   # Walk session, messages, trigger status & DebugInfo models
│       └── static/
│           ├── index.html   # Mobile web UI shell
│           ├── app.css      # Dark-mode & OLED pocket overlay styles
│           ├── app.js       # Main UI controller, GPS/ping loop & Pocket Mode
│           ├── api.js       # Authenticated fetch wrapper & client/token cookie sync
│           ├── audio.js     # Audio playback, MediaSession & silent keep-alive controller
│           ├── drawers.js   # Simulator & config drawer controllers
│           ├── state.js     # Reactive client state store & trigger-gate computation
│           └── telemetry.js # Frontend event buffering, beacon flush & GPS staleness tracking
└── tests/                   # Domain-organized pytest test suite
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

### 4. Static Analysis & Tests (`ruff`, `biome`, `ty`, `pytest`)
```bash
./deploy/check.sh
```

---

## Deploying to Google Cloud Run

The server is **fail-closed** by default: when `HEAR_ALLOW_UNAUTHENTICATED` is not set, all `/api/*` endpoints require `HEAR_AUTH_TOKEN` (via `x-hear-token` header or `SameSite=Strict` cookie).

Runtime data lives under `HEAR_DATA_DIR` (defaults to `./data` locally and `/data` in Docker). On Cloud Run, **Litestream** automatically restores and continuously replicates `${HEAR_DATA_DIR}/hear.db` to GCS (`gs://${HEAR_LITESTREAM_BUCKET}/hear.db`, defaulting to `<project>-hear-data`).

```bash
# One-time (or idempotent update): enable APIs, create Litestream GCS bucket, IAM & CI secrets
./deploy/setup_ci.sh

# Deploy directly from your machine:
./deploy/deploy.sh
```
