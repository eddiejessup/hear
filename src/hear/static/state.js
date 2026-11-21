export const appState = {
  currentConfig: null,
  selectedLensId: "",
  currentCoords: null,
  useSimulatedGps: false,
  watchId: null,
  isGenerating: false,
  isPlayingAudio: false,
  isPausedAudio: false,
  playbackSpeeds: [1.0, 1.15, 1.25, 1.4],
  speedIdx: 0,
  wakeLockSentinel: null,
  pingInFlight: false,
  lastPingSnapshot: null,
};

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
