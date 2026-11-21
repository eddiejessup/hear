export const GpsState = Object.freeze({
  WAITING: "waiting",
  ACTIVE: "active",
  SIMULATED: "simulated",
  STALE: "stale",
  ERROR: "error",
  UNSUPPORTED: "unsupported",
});

export const PlaybackState = Object.freeze({
  IDLE: "idle",
  PLAYING: "playing",
  PAUSED: "paused",
  ERROR: "error",
});

export const GenerationState = Object.freeze({
  IDLE: "idle",
  NARRATING: "narrating",
  CHATTING: "chatting",
});

export const WakeLockState = Object.freeze({
  OFF: "off",
  ACTIVE: "active",
  ERROR: "error",
});

export const NetworkState = Object.freeze({
  IDLE: "idle",
  OK: "ok",
  ERROR: "error",
  OFFLINE: "offline",
});

export const GPS_STALE_AFTER_MS = 25_000;

export const appState = {
  sessionId: null,
  currentConfig: null,
  selectedLensId: "",
  gps: {
    status: GpsState.WAITING,
    coords: null,
    lastFixAt: null,
    lastError: null,
    watchId: null,
    simulated: false,
  },
  playback: {
    status: PlaybackState.IDLE,
    speedIdx: 0,
    speeds: [1.0, 1.15, 1.25, 1.4],
    keepAliveActive: false,
    currentTitle: null,
    lastError: null,
  },
  generation: {
    status: GenerationState.IDLE,
    startedAt: null,
    reason: null,
  },
  trigger: {
    status: null,
    updatedAt: null,
  },
  network: {
    status:
      typeof navigator !== "undefined" && navigator.onLine === false
        ? NetworkState.OFFLINE
        : NetworkState.IDLE,
    pingInFlight: false,
    lastPingAt: null,
    lastPingSnapshot: null,
    lastError: null,
  },
  device: {
    visibility: typeof document !== "undefined" ? document.visibilityState : "visible",
    pocketMode: false,
    wakeLock: WakeLockState.OFF,
    wakeLockSentinel: null,
    wakeLockError: null,
  },
};

const listeners = new Set();

export function subscribeState(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function notifyStateChange(changeTag = "update") {
  for (const listener of listeners) {
    try {
      listener(appState, changeTag);
    } catch (err) {
      console.error("State subscriber error:", err);
    }
  }
}

export function patchSection(section, partial, changeTag = section) {
  Object.assign(appState[section], partial);
  notifyStateChange(changeTag);
}

export function setTopLevelState(partial, changeTag = "root") {
  Object.assign(appState, partial);
  notifyStateChange(changeTag);
}

export function isGenerating(state = appState) {
  return state.generation.status !== GenerationState.IDLE;
}

export function isPlayingAudio(state = appState) {
  return state.playback.status === PlaybackState.PLAYING;
}

export function isPausedAudio(state = appState) {
  return state.playback.status === PlaybackState.PAUSED;
}

export function isGpsStale(state = appState, nowMs = Date.now()) {
  if (state.gps.simulated || !state.gps.lastFixAt) return false;
  return nowMs - state.gps.lastFixAt > GPS_STALE_AFTER_MS;
}

export function getEffectiveGpsStatus(state = appState, nowMs = Date.now()) {
  if (state.gps.simulated) return GpsState.SIMULATED;
  if (
    state.gps.status === GpsState.UNSUPPORTED ||
    state.gps.status === GpsState.ERROR ||
    state.gps.status === GpsState.WAITING
  ) {
    return state.gps.status;
  }
  return isGpsStale(state, nowMs) ? GpsState.STALE : GpsState.ACTIVE;
}

export function getEstimatedCooldownRemaining(state = appState, nowMs = Date.now()) {
  const trig = state.trigger.status;
  if (!trig?.cooldown_remaining_s || trig.cooldown_remaining_s <= 0) return 0;
  const elapsedSinceUpdateS = state.trigger.updatedAt
    ? Math.max(0, (nowMs - state.trigger.updatedAt) / 1000)
    : 0;
  return Math.max(0, trig.cooldown_remaining_s - elapsedSinceUpdateS);
}

export function computeTriggerGate(state = appState, nowMs = Date.now()) {
  if (state.network.status === NetworkState.OFFLINE) {
    return { code: "offline", label: "Offline", tone: "error" };
  }
  if (isGenerating(state) || state.trigger.status?.is_generating) {
    const elapsedS = state.generation.startedAt
      ? Math.max(0, Math.floor((nowMs - state.generation.startedAt) / 1000))
      : null;
    const verb = state.generation.status === GenerationState.CHATTING ? "Replying" : "Researching";
    return {
      code: "generating",
      label: elapsedS !== null && elapsedS > 0 ? `${verb}… ${elapsedS}s` : `${verb}…`,
      tone: "active",
    };
  }
  if (isPlayingAudio(state)) {
    return { code: "playing", label: "Playing audio", tone: "active" };
  }
  const serverAckedPause = state.network.lastPingSnapshot?.isPausedAudio === true;
  const pauseHold = serverAckedPause
    ? Boolean(state.trigger.status?.is_paused_audio)
    : isPausedAudio(state) || Boolean(state.trigger.status?.is_paused_audio);
  if (pauseHold) {
    return { code: "paused_hold", label: "Paused (holding)", tone: "warn" };
  }

  const gpsStatus = getEffectiveGpsStatus(state, nowMs);
  if (gpsStatus === GpsState.UNSUPPORTED || gpsStatus === GpsState.ERROR) {
    return { code: "gps_error", label: "GPS unavailable", tone: "error" };
  }
  if (!state.gps.coords || gpsStatus === GpsState.WAITING) {
    return { code: "no_gps", label: "Waiting for GPS", tone: "warn" };
  }
  if (gpsStatus === GpsState.STALE) {
    const staleS = Math.floor((nowMs - state.gps.lastFixAt) / 1000);
    return { code: "gps_stale", label: `GPS stale (${staleS}s)`, tone: "warn" };
  }

  const trig = state.trigger.status;
  const autoEnabled =
    trig?.auto_trigger_enabled ?? state.currentConfig?.triggers?.auto_trigger_enabled ?? false;
  if (!autoEnabled) {
    return { code: "manual", label: "Manual", tone: "muted" };
  }

  const cooldownS = getEstimatedCooldownRemaining(state, nowMs);
  if (cooldownS > 0) {
    return { code: "cooldown", label: `Cooldown: ${Math.ceil(cooldownS)}s`, tone: "warn" };
  }

  if (trig?.should_trigger) {
    return { code: "ready", label: "Trigger ready", tone: "ok" };
  }

  return { code: "armed", label: "Armed", tone: "ok" };
}

export function formatAgeShort(tsMs, nowMs = Date.now()) {
  if (!tsMs) return "--";
  const diffS = Math.max(0, Math.floor((nowMs - tsMs) / 1000));
  if (diffS < 2) return "now";
  if (diffS < 60) return `${diffS}s`;
  const mins = Math.floor(diffS / 60);
  if (mins < 60) return `${mins}m`;
  return `${Math.floor(mins / 60)}h`;
}

export function approxDistanceMeters(lat1, lon1, lat2, lon2) {
  const r = 6371000;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * r * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

export function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function formatUptime(sec) {
  if (typeof sec !== "number" || sec < 0) return "--";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function parseNonNegativeFloat(raw, fallback) {
  const n = parseFloat(raw);
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}
