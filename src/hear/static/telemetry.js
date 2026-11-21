import { getAuthToken, getClientId, syncTokenCookie } from "./api.js";
import { GpsState, getEffectiveGpsStatus, subscribeState } from "./state.js";

export const TelemetryLevel = Object.freeze({
  DEBUG: "debug",
  INFO: "info",
  WARN: "warn",
  ERROR: "error",
});

export const FrontendEventType = Object.freeze({
  APP_INIT: "app_init",
  VISIBILITY_CHANGE: "visibility_change",
  PAGE_HIDE: "page_hide",
  POCKET_ENTER: "pocket_enter",
  POCKET_EXIT: "pocket_exit",
  WAKELOCK_ACQUIRED: "wakelock_acquired",
  WAKELOCK_RELEASED: "wakelock_released",
  WAKELOCK_ERROR: "wakelock_error",
  NETWORK_ONLINE: "network_online",
  NETWORK_OFFLINE: "network_offline",
  GPS_UNSUPPORTED: "gps_unsupported",
  GPS_ERROR: "gps_error",
  GPS_STALE: "gps_stale",
  GPS_RECOVERED: "gps_recovered",
  GPS_SIMULATED: "gps_simulated",
  AUDIO_PLAY: "audio_play",
  AUDIO_PAUSE: "audio_pause",
  AUDIO_ENDED: "audio_ended",
  AUDIO_ERROR: "audio_error",
  AUDIO_AUTOPLAY_BLOCKED: "audio_autoplay_blocked",
  TTS_PLAY: "tts_play",
  TTS_ENDED: "tts_ended",
  TTS_ERROR: "tts_error",
  KEEPALIVE_STARTED: "keepalive_started",
  KEEPALIVE_FAILED: "keepalive_failed",
  MEDIASESSION_ACTION: "mediasession_action",
  NARRATION_TRIGGERED: "narration_triggered",
  API_ERROR: "api_error",
  WINDOW_ERROR: "window_error",
  UNHANDLED_REJECTION: "unhandled_rejection",
});

const MAX_BUFFER_SIZE = 100;
const FLUSH_BATCH_THRESHOLD = 20;
const FLUSH_INTERVAL_MS = 15_000;

export function createTelemetryClient({
  state,
  endpoint = "/api/events",
  fetchFn = typeof fetch === "function" ? fetch.bind(globalThis) : null,
  sendBeaconFn = typeof navigator !== "undefined" && typeof navigator.sendBeacon === "function"
    ? navigator.sendBeacon.bind(navigator)
    : null,
  getToken = getAuthToken,
  getClient = getClientId,
} = {}) {
  const buffer = [];
  let flushInFlight = false;
  let flushTimerId = null;
  let totalSent = 0;
  let totalDropped = 0;
  let lastGpsProblemAt = null;
  let wasGpsStale = false;

  function buildHeaders() {
    const token = getToken();
    if (token && typeof document !== "undefined") {
      syncTokenCookie(token);
    }
    const headers = {
      "Content-Type": "application/json",
      "x-hear-client-id": getClient(),
    };
    if (token) headers["x-hear-token"] = token;
    return headers;
  }

  function track(eventType, payload = {}, level = TelemetryLevel.INFO) {
    const entry = {
      ts: Date.now() / 1000,
      level,
      session_id: state?.sessionId || null,
      event_type: eventType,
      payload: payload && typeof payload === "object" ? payload : { value: String(payload) },
    };
    if (buffer.length >= MAX_BUFFER_SIZE) {
      buffer.shift();
      totalDropped += 1;
    }
    buffer.push(entry);

    if (level === TelemetryLevel.ERROR || buffer.length >= FLUSH_BATCH_THRESHOLD) {
      flush();
    }
  }

  async function flush({ useBeacon = false } = {}) {
    if (buffer.length === 0) return 0;
    if (typeof navigator !== "undefined" && navigator.onLine === false) return 0;
    if (useBeacon) {
      const batch = buffer.splice(0, MAX_BUFFER_SIZE);
      const body = JSON.stringify({ events: batch });
      // Ensure cookies are fresh so sendBeacon carries auth + client_id
      getClient();
      const token = getToken();
      if (token && typeof document !== "undefined") syncTokenCookie(token);

      if (sendBeaconFn) {
        try {
          const blob = new Blob([body], { type: "application/json" });
          if (sendBeaconFn(endpoint, blob)) {
            totalSent += batch.length;
            return batch.length;
          }
        } catch (_e) {
          // Fall through to keepalive fetch
        }
      }
      if (fetchFn) {
        try {
          const pending = fetchFn(endpoint, {
            method: "POST",
            headers: buildHeaders(),
            body,
            keepalive: true,
          });
          if (pending && typeof pending.then === "function") {
            pending
              .then((res) => {
                if (res?.ok) {
                  totalSent += batch.length;
                } else if (!res || res.status >= 500 || res.status === 429) {
                  requeueFront(batch);
                } else {
                  totalDropped += batch.length;
                }
              })
              .catch(() => {
                requeueFront(batch);
              });
          }
          return batch.length;
        } catch (_e) {
          // Re-queue if unload dispatch fails synchronously
        }
      }
      requeueFront(batch);
      return 0;
    }

    if (flushInFlight || !fetchFn) return 0;

    flushInFlight = true;
    const batch = buffer.splice(0, MAX_BUFFER_SIZE);
    try {
      const res = await fetchFn(endpoint, {
        method: "POST",
        headers: buildHeaders(),
        body: JSON.stringify({ events: batch }),
      });
      if (res.ok) {
        totalSent += batch.length;
        return batch.length;
      }
      // Drop permanently rejected client batches (401/422) rather than looping forever
      if (res.status >= 500 || res.status === 429) {
        requeueFront(batch);
      } else {
        totalDropped += batch.length;
      }
      return 0;
    } catch (_err) {
      requeueFront(batch);
      return 0;
    } finally {
      flushInFlight = false;
    }
  }

  function requeueFront(batch) {
    const combined = [...batch, ...buffer];
    if (combined.length > MAX_BUFFER_SIZE) {
      const overflow = combined.length - MAX_BUFFER_SIZE;
      totalDropped += overflow;
      combined.splice(0, overflow);
    }
    buffer.length = 0;
    buffer.push(...combined);
  }

  function checkGpsStaleness(nowMs = Date.now()) {
    if (!state) return;
    const effective = getEffectiveGpsStatus(state, nowMs);
    if (effective === GpsState.STALE && !wasGpsStale) {
      wasGpsStale = true;
      lastGpsProblemAt = state.gps.lastFixAt || nowMs;
      track(
        FrontendEventType.GPS_STALE,
        {
          gap_seconds: state.gps.lastFixAt
            ? Math.round((nowMs - state.gps.lastFixAt) / 1000)
            : null,
          visibility: state.device.visibility,
          pocket_mode: state.device.pocketMode,
        },
        TelemetryLevel.WARN
      );
    } else if (effective === GpsState.ACTIVE && wasGpsStale) {
      wasGpsStale = false;
      const gapS = lastGpsProblemAt ? Math.round((nowMs - lastGpsProblemAt) / 1000) : null;
      lastGpsProblemAt = null;
      track(
        FrontendEventType.GPS_RECOVERED,
        {
          gap_seconds: gapS,
          accuracy: state.gps.coords?.accuracy ? Math.round(state.gps.coords.accuracy) : null,
        },
        TelemetryLevel.INFO
      );
    }
  }

  function attachStateObserver() {
    return subscribeState((cur, tag) => {
      switch (tag) {
        case "visibility_change":
          track(FrontendEventType.VISIBILITY_CHANGE, {
            state: cur.device.visibility,
            pocket_mode: cur.device.pocketMode,
            playback: cur.playback.status,
          });
          if (cur.device.visibility === "hidden") {
            flush({ useBeacon: true });
          }
          break;
        case "pocket_enter":
          track(FrontendEventType.POCKET_ENTER);
          break;
        case "pocket_exit":
          track(FrontendEventType.POCKET_EXIT);
          break;
        case "wakelock_acquired":
          track(FrontendEventType.WAKELOCK_ACQUIRED);
          break;
        case "wakelock_released":
          track(FrontendEventType.WAKELOCK_RELEASED, {
            visibility: cur.device.visibility,
            pocket_mode: cur.device.pocketMode,
          });
          break;
        case "wakelock_error":
          track(
            FrontendEventType.WAKELOCK_ERROR,
            { error: cur.device.wakeLockError },
            TelemetryLevel.WARN
          );
          break;
        case "network_online":
          track(FrontendEventType.NETWORK_ONLINE);
          flush();
          break;
        case "network_offline":
          track(FrontendEventType.NETWORK_OFFLINE, {}, TelemetryLevel.WARN);
          break;
        case "gps_unsupported":
          track(FrontendEventType.GPS_UNSUPPORTED, {}, TelemetryLevel.ERROR);
          break;
        case "gps_error":
          wasGpsStale = true;
          if (!lastGpsProblemAt) lastGpsProblemAt = Date.now();
          track(
            FrontendEventType.GPS_ERROR,
            { error: cur.gps.lastError, visibility: cur.device.visibility },
            TelemetryLevel.WARN
          );
          break;
        case "gps_fix":
          checkGpsStaleness();
          break;
        case "gps_simulated":
          wasGpsStale = false;
          lastGpsProblemAt = null;
          track(FrontendEventType.GPS_SIMULATED, {
            lat: cur.gps.coords ? Number(cur.gps.coords.lat.toFixed(5)) : null,
            lon: cur.gps.coords ? Number(cur.gps.coords.lon.toFixed(5)) : null,
          });
          break;
        case "audio_play":
          track(FrontendEventType.AUDIO_PLAY, { title: cur.playback.currentTitle });
          break;
        case "audio_pause":
          track(FrontendEventType.AUDIO_PAUSE, { title: cur.playback.currentTitle });
          break;
        case "audio_ended":
          track(FrontendEventType.AUDIO_ENDED, { title: cur.playback.currentTitle });
          break;
        case "audio_error":
          track(
            FrontendEventType.AUDIO_ERROR,
            { error: cur.playback.lastError, title: cur.playback.currentTitle },
            TelemetryLevel.ERROR
          );
          break;
        case "audio_autoplay_blocked":
          track(
            FrontendEventType.AUDIO_AUTOPLAY_BLOCKED,
            { error: cur.playback.lastError, title: cur.playback.currentTitle },
            TelemetryLevel.WARN
          );
          break;
        case "tts_play":
          track(FrontendEventType.TTS_PLAY);
          break;
        case "tts_ended":
          track(FrontendEventType.TTS_ENDED);
          break;
        case "tts_error":
          track(
            FrontendEventType.TTS_ERROR,
            { error: cur.playback.lastError },
            TelemetryLevel.WARN
          );
          break;
        case "keepalive_started":
          track(FrontendEventType.KEEPALIVE_STARTED);
          break;
        default:
          break;
      }
    });
  }

  function installGlobalHandlers() {
    window.addEventListener("error", (ev) => {
      track(
        FrontendEventType.WINDOW_ERROR,
        {
          message: ev.message || "unknown_error",
          filename: ev.filename || null,
          lineno: ev.lineno || null,
          colno: ev.colno || null,
        },
        TelemetryLevel.ERROR
      );
    });

    window.addEventListener("unhandledrejection", (ev) => {
      const reason = ev.reason;
      track(
        FrontendEventType.UNHANDLED_REJECTION,
        {
          reason:
            reason instanceof Error ? `${reason.name}: ${reason.message}` : String(reason || ""),
        },
        TelemetryLevel.ERROR
      );
    });

    window.addEventListener("pagehide", () => {
      track(FrontendEventType.PAGE_HIDE, {
        visibility: state?.device?.visibility || null,
        pocket_mode: state?.device?.pocketMode || false,
      });
      flush({ useBeacon: true });
    });
  }

  function start() {
    attachStateObserver();
    installGlobalHandlers();
    track(FrontendEventType.APP_INIT, {
      standalone:
        typeof window !== "undefined" &&
        (window.matchMedia?.("(display-mode: standalone)")?.matches ||
          window.navigator?.standalone === true),
      has_geolocation: typeof navigator !== "undefined" && "geolocation" in navigator,
      has_wakelock: typeof navigator !== "undefined" && "wakeLock" in navigator,
      has_mediasession: typeof navigator !== "undefined" && "mediaSession" in navigator,
      user_agent: typeof navigator !== "undefined" ? navigator.userAgent.slice(0, 160) : "",
    });
    if (flushTimerId === null) {
      flushTimerId = setInterval(() => flush(), FLUSH_INTERVAL_MS);
    }
  }

  function getStats() {
    return {
      buffered: buffer.length,
      sent: totalSent,
      dropped: totalDropped,
    };
  }

  return {
    track,
    flush,
    checkGpsStaleness,
    start,
    getStats,
  };
}
