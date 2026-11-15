# Hear — Free-Roam Historical Walking Companion

A personal, location-aware historical audio companion powered by **Gemini** (`gemini-3.8-flash` / `gemini-4-argon` with Google Search grounding + `gemini-3.8-flash-tts`).

## Project Structure
```text
hear/
├── config.yaml              # Hot-reloaded lenses, system prompt, model/voice, and trigger rules
├── pyproject.toml           # Python 3.15 project config (uv, ruff, ty)
└── src/
    └── hear/
        ├── __init__.py
        ├── app.py           # FastAPI application, endpoints, and CLI entry point
        ├── config.py        # Hot-reloading config.yaml manager
        ├── gemini.py        # Gemini generateContent (with Google Search) & TTS client
        ├── geo.py           # Haversine distance, compass bearings, Nominatim & Overpass lookups
        ├── models.py        # Pydantic & TypedDict schemas
        ├── paths.py         # Project paths & directory initialization
        ├── session.py       # WalkSession state machine, lens sampling & trigger heuristics
        └── static/
            └── index.html   # Mobile web UI
```

## Quick Start (with `uv`)

### 1. Sync environment (Python 3.15)
```bash
uv sync
```

### 2. Set your Gemini API key
Export `GEMINI_API_KEY` in your shell or create a `.env` file in the project root:
```bash
echo 'GEMINI_API_KEY="your-api-key-here"' > .env
```

### 3. Run the server
```bash
uv run hear
# or directly with uvicorn:
uv run uvicorn hear.app:app --reload --host 127.0.0.1 --port 8000
```
- **Desktop testing:** Open `http://localhost:8000` in your browser and click **Sim** in the top bar to teleport to preset coordinates or simulate walking 150m at a time.

### 4. Static Analysis (`ruff` & `ty`)
```bash
uv run ruff check .
uv run ty check
```

### 5. Connect from your Pixel over Tailscale (HTTPS)
Mobile Chrome requires an `https://` origin (or `localhost`) to enable `navigator.geolocation` and `navigator.wakeLock`. With Tailscale installed on your desktop and your Pixel:
```bash
tailscale serve --bg 8000
```
Then open the resulting `https://<your-desktop>.<your-tailnet>.ts.net` URL in Chrome on your Pixel (and optionally tap **Add to Home screen**).
