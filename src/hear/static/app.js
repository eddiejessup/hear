import { apiFetch } from "./api.js";
import { createAudioController } from "./audio.js";
import { createDrawersController, renderDebugInfo, updateAutoTriggerButton } from "./drawers.js";
import {
  approxDistanceMeters,
  appState,
  computeTriggerGate,
  escapeHtml,
  formatAgeShort,
  GenerationState,
  GpsState,
  getEffectiveGpsStatus,
  getEstimatedCooldownRemaining,
  isGenerating,
  isPausedAudio,
  isPlayingAudio,
  NetworkState,
  patchSection,
  setTopLevelState,
  subscribeState,
  WakeLockState,
} from "./state.js";
import { createTelemetryClient, FrontendEventType, TelemetryLevel } from "./telemetry.js";

const mainAudio = document.getElementById("mainAudio");
const audioBar = document.getElementById("audioBar");
const speedBtn = document.getElementById("speedBtn");

const telemetry = createTelemetryClient({ state: appState });

const audio = createAudioController({
  mainAudio,
  audioBar,
  speedBtn,
  state: appState,
  onAudioEnded: () => sendPing(true),
  onNextTrack: (reason) => triggerNarration(reason),
  onMediaSessionAction: (action) =>
    telemetry.track(FrontendEventType.MEDIASESSION_ACTION, { action }),
  onKeepAliveError: (error) =>
    telemetry.track(FrontendEventType.KEEPALIVE_FAILED, { error }, TelemetryLevel.WARN),
});

const drawers = createDrawersController({
  state: appState,
  audio,
  renderLensesStrip,
  sendPing,
  startGpsWatch,
});

// ---------------------------------------------------------------------------
// Live State UI Rendering (driven by state subscriptions + 1s ticker)
// ---------------------------------------------------------------------------
function setBadge(el, text, tone = "muted") {
  if (!el) return;
  el.textContent = text;
  const clickable = el.classList.contains("clickable") ? " clickable" : "";
  const toneClass = tone && tone !== "muted" ? ` tone-${tone}` : "";
  el.className = `badge${clickable}${toneClass}`;
}

function renderStateUI() {
  const nowMs = Date.now();
  const gpsStatus = getEffectiveGpsStatus(appState, nowMs);
  const coords = appState.gps.coords;
  const trig = appState.trigger.status;
  const gate = computeTriggerGate(appState, nowMs);

  // 1. GPS indicator dot & location title
  const gpsDot = document.getElementById("gpsDot");
  if (gpsStatus === GpsState.ACTIVE || gpsStatus === GpsState.SIMULATED) {
    gpsDot.className = "gps-dot active";
  } else if (gpsStatus === GpsState.ERROR || gpsStatus === GpsState.UNSUPPORTED) {
    gpsDot.className = "gps-dot error";
  } else {
    gpsDot.className = "gps-dot";
  }

  const locLabel = document.getElementById("locLabel");
  if (coords) {
    locLabel.textContent =
      coords.short_label || `${coords.lat.toFixed(5)}, ${coords.lon.toFixed(5)}`;
  } else if (gpsStatus === GpsState.UNSUPPORTED) {
    locLabel.textContent = "Geolocation not supported in this browser";
  } else if (gpsStatus === GpsState.ERROR && appState.gps.lastError) {
    locLabel.textContent = `GPS waiting (${appState.gps.lastError}) — or use Sim`;
  } else {
    locLabel.textContent = "Waiting for GPS lock…";
  }

  // 2. GPS freshness badge
  const gpsBadge = document.getElementById("gpsStatusBadge");
  if (gpsStatus === GpsState.SIMULATED) {
    setBadge(gpsBadge, "GPS: sim", "active");
  } else if (gpsStatus === GpsState.ACTIVE) {
    const age = formatAgeShort(appState.gps.lastFixAt, nowMs);
    const acc = coords?.accuracy ? ` ±${Math.round(coords.accuracy)}m` : "";
    setBadge(gpsBadge, `GPS: ${age}${acc}`, "ok");
  } else if (gpsStatus === GpsState.STALE) {
    const age = formatAgeShort(appState.gps.lastFixAt, nowMs);
    setBadge(gpsBadge, `GPS stale (${age})`, "warn");
  } else if (gpsStatus === GpsState.ERROR || gpsStatus === GpsState.UNSUPPORTED) {
    setBadge(gpsBadge, "GPS: err", "error");
  } else {
    setBadge(gpsBadge, "GPS: waiting", "warn");
  }

  // 3. Distance badge
  const distBadge = document.getElementById("distBadge");
  const minDist = Math.round(
    trig?.min_distance_m ?? appState.currentConfig?.triggers?.min_distance_meters ?? 120
  );
  const distVal =
    trig?.distance_moved_m !== null && trig?.distance_moved_m !== undefined
      ? `${Math.round(trig.distance_moved_m)}m`
      : "0m";
  setBadge(distBadge, `Moved: ${distVal} / ${minDist}m`, "muted");

  // 4. Primary trigger/activity gate badge (clickable to inspect full state)
  const gateBadge = document.getElementById("statusGateBadge");
  setBadge(gateBadge, gate.label, gate.tone);

  // 5. Sync / device health badge
  const syncBadge = document.getElementById("syncBadge");
  const wlSuffix = appState.device.wakeLock === WakeLockState.ACTIVE ? " • WL" : "";
  if (appState.network.status === NetworkState.OFFLINE) {
    setBadge(syncBadge, `Offline${wlSuffix}`, "error");
  } else if (appState.network.status === NetworkState.ERROR) {
    setBadge(syncBadge, `Sync err${wlSuffix}`, "error");
  } else if (appState.network.lastPingAt) {
    const pingAge = formatAgeShort(appState.network.lastPingAt, nowMs);
    setBadge(syncBadge, `Sync: ${pingAge}${wlSuffix}`, "muted");
  } else {
    setBadge(syncBadge, `Sync: --${wlSuffix}`, "muted");
  }

  // 6. Pocket Mode status line
  const pocketStatus = document.getElementById("pocketStatus");
  if (pocketStatus) {
    const place = coords?.short_label || "No GPS";
    pocketStatus.textContent = `${place} • Moved ${distVal} • ${gate.label}`;
  }

  // 7. Collapsible State Inspector panel (only render DOM when expanded)
  const panel = document.getElementById("stateInspectorPanel");
  if (panel?.classList.contains("open")) {
    const coordStr = coords
      ? `${coords.lat.toFixed(5)}, ${coords.lon.toFixed(5)}${coords.accuracy ? ` (±${Math.round(coords.accuracy)}m)` : ""}`
      : "--";
    const cooldownS = getEstimatedCooldownRemaining(appState, nowMs);
    const silenceStr =
      trig?.silence_elapsed_s !== null && trig?.silence_elapsed_s !== undefined
        ? `${Math.round(trig.silence_elapsed_s)}s / ${Math.round(trig.max_silence_s)}s`
        : "--";
    const speedRate = appState.playback.speeds[appState.playback.speedIdx];
    const items = [
      ["GPS", `${gpsStatus} • fix ${formatAgeShort(appState.gps.lastFixAt, nowMs)}`],
      ["Coords", coordStr],
      ["Gate", `${gate.code} (${gate.label})`],
      ["Cooldown / Silence", `${Math.ceil(cooldownS)}s • ${silenceStr}`],
      [
        "Playback",
        `${appState.playback.status} • ${speedRate}x • KA:${appState.playback.keepAliveActive ? "on" : "off"}`,
      ],
      [
        "Generation",
        appState.generation.status === GenerationState.IDLE
          ? "idle"
          : `${appState.generation.status} (${appState.generation.reason || "manual"})`,
      ],
      [
        "Device",
        `vis:${appState.device.visibility} • wl:${appState.device.wakeLock} • pocket:${appState.device.pocketMode ? "on" : "off"}`,
      ],
      [
        "Network",
        `${appState.network.status} • ping ${formatAgeShort(appState.network.lastPingAt, nowMs)}${appState.network.lastError ? ` (${appState.network.lastError})` : ""}`,
      ],
      ["Session", appState.sessionId || "--"],
      [
        "Telemetry",
        (() => {
          const s = telemetry.getStats();
          return `q:${s.buffered} • sent:${s.sent}${s.dropped ? ` • drop:${s.dropped}` : ""}`;
        })(),
      ],
    ];
    if (appState.gps.lastError) {
      items.push(["GPS Error", appState.gps.lastError]);
    }
    if (appState.playback.lastError) {
      items.push(["Audio Error", appState.playback.lastError]);
    }
    const canPatchInPlace =
      panel.childElementCount === items.length &&
      items.every(([label], idx) => panel.children[idx]?.dataset.label === label);
    if (canPatchInPlace) {
      items.forEach(([, val], idx) => {
        const valEl = panel.children[idx].querySelector(".state-item-value");
        if (valEl && valEl.textContent !== val) {
          valEl.textContent = val;
          valEl.title = val;
        }
      });
    } else {
      panel.innerHTML = items
        .map(
          ([label, val]) => `
        <div class="state-item" data-label="${escapeHtml(label)}">
          <span class="state-item-label">${escapeHtml(label)}</span>
          <span class="state-item-value" title="${escapeHtml(val)}">${escapeHtml(val)}</span>
        </div>`
        )
        .join("");
    }
  }
}

// ---------------------------------------------------------------------------
// Initialization & Event Wiring
// ---------------------------------------------------------------------------
function bindDomEvents() {
  document.getElementById("versionBadge").addEventListener("click", drawers.openSimDrawer);
  document.getElementById("autoToggleBtn").addEventListener("click", drawers.toggleAutoTrigger);
  document.getElementById("newWalkBtn").addEventListener("click", drawers.startNewWalk);
  document.getElementById("pocketBtn").addEventListener("click", enterPocketMode);
  document.getElementById("openSimBtn").addEventListener("click", drawers.openSimDrawer);
  document.getElementById("openConfigBtn").addEventListener("click", drawers.openConfigDrawer);

  document.getElementById("statusGateBadge").addEventListener("click", (event) => {
    const isOpen = document.getElementById("stateInspectorPanel").classList.toggle("open");
    event.currentTarget.setAttribute("aria-expanded", String(isOpen));
    renderStateUI();
  });

  document.getElementById("narrateBtn").addEventListener("click", () => triggerNarration("manual"));
  speedBtn.addEventListener("click", audio.cycleSpeed);

  document.getElementById("chatForm").addEventListener("submit", sendChat);
  document.getElementById("micBtn").addEventListener("click", toggleDictation);

  document.getElementById("pocketOverlay").addEventListener("dblclick", exitPocketMode);
  document.getElementById("pocketNarrateBtn").addEventListener("click", (event) => {
    event.stopPropagation();
    triggerNarration("pocket_button");
  });

  // Delegated click handler for play buttons inside the message feed
  document.getElementById("feed").addEventListener("click", (event) => {
    const btn = event.target.closest(".play-clip-btn");
    if (!btn) return;
    if (btn.dataset.audioUrl) {
      audio.playAudioUrl(btn.dataset.audioUrl, btn.dataset.title, btn.dataset.subtitle);
    } else if (btn.dataset.text) {
      audio.speakBrowserTts(btn.dataset.text);
    }
  });

  // Simulator drawer events
  const simBackdrop = document.getElementById("simBackdrop");
  simBackdrop.addEventListener("click", (event) => {
    if (event.target === simBackdrop) drawers.closeSimDrawer();
  });
  document.getElementById("closeSimBtn").addEventListener("click", drawers.closeSimDrawer);
  document.querySelectorAll("[data-sim-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const [lat, lon] = btn.dataset.simPreset.split(",").map(Number);
      drawers.setSimPreset(lat, lon);
    });
  });
  document.getElementById("applySimBtn").addEventListener("click", drawers.applySimCoords);
  document
    .getElementById("nudgeNorthBtn")
    .addEventListener("click", () => drawers.nudgeSim(150, 0));
  document.getElementById("nudgeEastBtn").addEventListener("click", () => drawers.nudgeSim(0, 150));
  document.getElementById("resumeGpsBtn").addEventListener("click", drawers.resumeRealGps);
  document.getElementById("simNewWalkBtn").addEventListener("click", drawers.startNewWalk);

  // Config drawer events
  const configBackdrop = document.getElementById("configBackdrop");
  configBackdrop.addEventListener("click", (event) => {
    if (event.target === configBackdrop) drawers.closeConfigDrawer();
  });
  document.getElementById("closeConfigBtn").addEventListener("click", drawers.closeConfigDrawer);
  document.getElementById("addLensBtn").addEventListener("click", drawers.addLensRow);
  document.getElementById("saveConfigBtn").addEventListener("click", drawers.saveConfigToServer);
  document.getElementById("cfgLensesList").addEventListener("click", (event) => {
    const removeBtn = event.target.closest(".lens-remove-btn");
    if (removeBtn) {
      removeBtn.closest(".lens-editor-item")?.remove();
    }
  });

  // Device & network lifecycle listeners
  document.addEventListener("visibilitychange", () => {
    const visibility = document.visibilityState;
    patchSection("device", { visibility }, "visibility_change");
    if (visibility === "visible" && appState.device.pocketMode) {
      requestScreenWakeLock();
    }
  });
  window.addEventListener("online", () => {
    patchSection("network", { status: NetworkState.OK, lastError: null }, "network_online");
    sendPing(false);
  });
  window.addEventListener("offline", () => {
    patchSection("network", { status: NetworkState.OFFLINE }, "network_offline");
  });
}

async function init() {
  subscribeState(() => renderStateUI());
  bindDomEvents();
  audio.setupAudioEvents();
  audio.setupMediaSession();
  telemetry.start();
  renderStateUI();
  await fetchInitialState();
  telemetry.flush();
  startGpsWatch();
  setInterval(periodicPing, 6000);
  setInterval(() => {
    telemetry.checkGpsStaleness();
    if (document.visibilityState !== "hidden") {
      renderStateUI();
    }
  }, 1000);
}

async function fetchInitialState() {
  try {
    const res = await apiFetch("/api/state");
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      patchSection(
        "network",
        { status: NetworkState.ERROR, lastError: `HTTP ${res.status}` },
        "state_error"
      );
      telemetry.track(
        FrontendEventType.API_ERROR,
        { endpoint: "/api/state", status_code: res.status, detail: err.detail || res.statusText },
        TelemetryLevel.ERROR
      );
      addSystemBanner(`⚠️ ${err.detail || res.statusText}`);
      return;
    }
    const data = await res.json();
    setTopLevelState({ sessionId: data.session_id, currentConfig: data.config }, "initial_state");
    patchSection("network", { status: NetworkState.OK, lastError: null }, "initial_state");
    patchSection(
      "trigger",
      { status: data.trigger_status, updatedAt: Date.now() },
      "initial_state"
    );
    renderLensesStrip(appState.currentConfig.lenses || []);
    updateAutoTriggerButton(appState.currentConfig.triggers?.auto_trigger_enabled ?? true);
    renderDebugInfo(data.debug_info, data.session_id);

    if (!data.has_api_key) {
      addSystemBanner(
        "⚠️ GEMINI_API_KEY is not set on the server. Set GEMINI_API_KEY in your environment or .env file."
      );
    }

    if (data.current_location) {
      patchSection("gps", { coords: data.current_location }, "initial_location");
    }

    if (data.messages && data.messages.length > 0) {
      document.getElementById("emptyState").style.display = "none";
      for (const msg of data.messages) {
        appendMessageCard(msg, false);
      }
    }
  } catch (err) {
    console.error("Failed to load initial state:", err);
    patchSection(
      "network",
      {
        status: navigator.onLine === false ? NetworkState.OFFLINE : NetworkState.ERROR,
        lastError: err?.message || "init_failed",
      },
      "state_error"
    );
    telemetry.track(
      FrontendEventType.API_ERROR,
      { endpoint: "/api/state", error: err?.message || "init_failed" },
      TelemetryLevel.ERROR
    );
  }
}

// ---------------------------------------------------------------------------
// GPS Tracking & Server Ping Loop
// ---------------------------------------------------------------------------
function startGpsWatch() {
  if (!navigator.geolocation) {
    patchSection(
      "gps",
      { status: GpsState.UNSUPPORTED, lastError: "Geolocation unsupported" },
      "gps_unsupported"
    );
    return;
  }
  if (appState.gps.watchId !== null) {
    navigator.geolocation.clearWatch(appState.gps.watchId);
  }
  const watchId = navigator.geolocation.watchPosition(
    (pos) => {
      if (appState.gps.simulated) return;
      patchSection(
        "gps",
        {
          status: GpsState.ACTIVE,
          lastFixAt: Date.now(),
          lastError: null,
          coords: {
            lat: pos.coords.latitude,
            lon: pos.coords.longitude,
            accuracy: pos.coords.accuracy,
            heading: pos.coords.heading,
            speed: pos.coords.speed,
            short_label: appState.gps.coords?.short_label,
          },
        },
        "gps_fix"
      );
      sendPing(false);
    },
    (err) => {
      if (appState.gps.simulated) return;
      console.warn("GPS watch error:", err.message);
      patchSection("gps", { status: GpsState.ERROR, lastError: err.message }, "gps_error");
    },
    {
      enableHighAccuracy: true,
      maximumAge: 5000,
      timeout: 15000,
    }
  );
  patchSection("gps", { watchId }, "gps_watch_started");
}

async function periodicPing() {
  const coords = appState.gps.coords;
  if (!coords || isGenerating()) return;
  const gpsStatus = getEffectiveGpsStatus(appState);
  if (gpsStatus !== GpsState.ACTIVE && gpsStatus !== GpsState.SIMULATED) return;
  const autoEnabled = appState.currentConfig?.triggers?.auto_trigger_enabled ?? false;
  const playing = isPlayingAudio();
  const paused = isPausedAudio();
  const snap = appState.network.lastPingSnapshot;
  if (
    !autoEnabled &&
    snap &&
    snap.isPlayingAudio === playing &&
    snap.isPausedAudio === paused &&
    approxDistanceMeters(coords.lat, coords.lon, snap.lat, snap.lon) < 3.0
  ) {
    return;
  }
  await sendPing(false);
}

async function sendPing(audioJustEnded = false) {
  const coords = appState.gps.coords;
  if (!coords) return;
  if (appState.network.pingInFlight && !audioJustEnded) return;
  appState.network.pingInFlight = true;
  const playing = isPlayingAudio();
  const paused = isPausedAudio();
  try {
    const res = await apiFetch("/api/ping", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lat: coords.lat,
        lon: coords.lon,
        accuracy: coords.accuracy ?? null,
        heading: coords.heading ?? null,
        speed: coords.speed ?? null,
        is_playing_audio: playing,
        is_paused_audio: paused,
        audio_just_ended: audioJustEnded,
      }),
    });
    if (!res.ok) {
      patchSection(
        "network",
        { status: NetworkState.ERROR, lastError: `Ping HTTP ${res.status}` },
        "ping_error"
      );
      telemetry.track(
        FrontendEventType.API_ERROR,
        { endpoint: "/api/ping", status_code: res.status },
        TelemetryLevel.WARN
      );
      return;
    }
    const data = await res.json();
    const nowMs = Date.now();
    if (data.location?.short_label && appState.gps.coords) {
      patchSection(
        "gps",
        { coords: { ...appState.gps.coords, short_label: data.location.short_label } },
        "gps_label"
      );
    }
    patchSection(
      "network",
      {
        status: NetworkState.OK,
        lastPingAt: nowMs,
        lastError: null,
        lastPingSnapshot: {
          lat: coords.lat,
          lon: coords.lon,
          isPlayingAudio: playing,
          isPausedAudio: paused,
        },
      },
      "ping_ok"
    );
    patchSection("trigger", { status: data.trigger_status, updatedAt: nowMs }, "ping_ok");

    const liveGpsStatus = getEffectiveGpsStatus(appState, nowMs);
    const gpsReady = liveGpsStatus === GpsState.ACTIVE || liveGpsStatus === GpsState.SIMULATED;
    if (data.trigger_status?.should_trigger && gpsReady && !isGenerating() && !isPlayingAudio()) {
      triggerNarration(data.trigger_status.trigger_reason || "auto");
    }
  } catch (err) {
    console.warn("Ping failed:", err);
    patchSection(
      "network",
      {
        status: navigator.onLine === false ? NetworkState.OFFLINE : NetworkState.ERROR,
        lastError: err?.message || "network_error",
      },
      "ping_error"
    );
    telemetry.track(
      FrontendEventType.API_ERROR,
      { endpoint: "/api/ping", error: err?.message || "network_error" },
      TelemetryLevel.WARN
    );
  } finally {
    appState.network.pingInFlight = false;
  }
}

// ---------------------------------------------------------------------------
// Narration & Chat Steering
// ---------------------------------------------------------------------------
async function triggerNarration(reason = "manual") {
  if (isGenerating()) return;
  const coords = appState.gps.coords;
  if (!coords) {
    alert(
      "Waiting for GPS coordinates (or open 'Sim' in the top bar to test with simulated coordinates)."
    );
    return;
  }

  audio.ensureSilentKeepAlive();
  telemetry.track(FrontendEventType.NARRATION_TRIGGERED, {
    reason,
    lens_id: appState.selectedLensId || null,
    gps_status: getEffectiveGpsStatus(appState),
    visibility: appState.device.visibility,
  });
  patchSection(
    "generation",
    { status: GenerationState.NARRATING, startedAt: Date.now(), reason },
    "narration_start"
  );
  const btn = document.getElementById("narrateBtn");
  btn.disabled = true;
  btn.innerHTML = `<span>⏳ Researching &amp; generating audio…</span>`;

  try {
    const res = await apiFetch("/api/narrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lat: coords.lat,
        lon: coords.lon,
        heading: coords.heading ?? null,
        lens_id: appState.selectedLensId || null,
        reason,
      }),
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      if (res.status !== 409) {
        patchSection(
          "network",
          { status: NetworkState.ERROR, lastError: `Narrate HTTP ${res.status}` },
          "narration_error"
        );
        telemetry.track(
          FrontendEventType.API_ERROR,
          { endpoint: "/api/narrate", status_code: res.status, reason },
          TelemetryLevel.ERROR
        );
        addSystemBanner(`Error: ${data.detail || res.statusText}`);
      }
      return;
    }

    const emptyEl = document.getElementById("emptyState");
    if (emptyEl) emptyEl.style.display = "none";
    if (data.message?.location?.short_label && appState.gps.coords) {
      patchSection(
        "gps",
        { coords: { ...appState.gps.coords, short_label: data.message.location.short_label } },
        "gps_label"
      );
    }
    patchSection("network", { status: NetworkState.OK, lastError: null }, "narration_ok");
    patchSection("trigger", { status: data.trigger_status, updatedAt: Date.now() }, "narration_ok");
    appendMessageCard(data.message, true);
  } catch (err) {
    patchSection(
      "network",
      {
        status: navigator.onLine === false ? NetworkState.OFFLINE : NetworkState.ERROR,
        lastError: err?.message || "network_error",
      },
      "narration_error"
    );
    telemetry.track(
      FrontendEventType.API_ERROR,
      { endpoint: "/api/narrate", error: err?.message || "network_error", reason },
      TelemetryLevel.ERROR
    );
    addSystemBanner(`Network error: ${err.message}`);
  } finally {
    patchSection(
      "generation",
      { status: GenerationState.IDLE, startedAt: null, reason: null },
      "narration_end"
    );
    btn.disabled = false;
    btn.innerHTML = `<span>🎙️ Tell me about here</span>`;
  }
}

async function sendChat(event) {
  event.preventDefault();
  const input = document.getElementById("chatInput");
  const text = input.value.trim();
  if (!text || isGenerating()) return;

  audio.ensureSilentKeepAlive();
  input.value = "";
  patchSection(
    "generation",
    { status: GenerationState.CHATTING, startedAt: Date.now(), reason: "chat" },
    "chat_start"
  );
  document.getElementById("sendBtn").disabled = true;

  const emptyEl = document.getElementById("emptyState");
  if (emptyEl) emptyEl.style.display = "none";
  const userCard = appendMessageCard(
    {
      role: "user",
      text,
      location_label: appState.gps.coords?.short_label,
    },
    false
  );

  try {
    const res = await apiFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        lat: appState.gps.coords?.lat ?? null,
        lon: appState.gps.coords?.lon ?? null,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      userCard?.remove();
      input.value = text;
      patchSection(
        "network",
        { status: NetworkState.ERROR, lastError: `Chat HTTP ${res.status}` },
        "chat_error"
      );
      telemetry.track(
        FrontendEventType.API_ERROR,
        { endpoint: "/api/chat", status_code: res.status },
        TelemetryLevel.ERROR
      );
      addSystemBanner(`Error: ${data.detail || res.statusText}`);
      return;
    }
    if (data.assistant_message?.location?.short_label && appState.gps.coords) {
      patchSection(
        "gps",
        {
          coords: {
            ...appState.gps.coords,
            short_label: data.assistant_message.location.short_label,
          },
        },
        "gps_label"
      );
    }
    patchSection("network", { status: NetworkState.OK, lastError: null }, "chat_ok");
    patchSection("trigger", { status: data.trigger_status, updatedAt: Date.now() }, "chat_ok");
    appendMessageCard(data.assistant_message, true);
  } catch (err) {
    userCard?.remove();
    input.value = text;
    patchSection(
      "network",
      {
        status: navigator.onLine === false ? NetworkState.OFFLINE : NetworkState.ERROR,
        lastError: err?.message || "network_error",
      },
      "chat_error"
    );
    telemetry.track(
      FrontendEventType.API_ERROR,
      { endpoint: "/api/chat", error: err?.message || "network_error" },
      TelemetryLevel.ERROR
    );
    addSystemBanner(`Chat error: ${err.message}`);
  } finally {
    patchSection(
      "generation",
      { status: GenerationState.IDLE, startedAt: null, reason: null },
      "chat_end"
    );
    document.getElementById("sendBtn").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// UI Rendering
// ---------------------------------------------------------------------------
function renderLensesStrip(lenses) {
  const strip = document.getElementById("lensStrip");
  strip.innerHTML = "";

  const randomChip = document.createElement("button");
  randomChip.className = `lens-chip ${appState.selectedLensId === "" ? "selected" : ""}`;
  randomChip.textContent = "🎲 Random Pool";
  randomChip.addEventListener("click", () => {
    setTopLevelState({ selectedLensId: "" }, "lens_selected");
    renderLensesStrip(lenses);
  });
  strip.appendChild(randomChip);

  for (const lens of lenses) {
    if (lens.enabled === false) continue;
    const chip = document.createElement("button");
    chip.className = `lens-chip ${appState.selectedLensId === lens.id ? "selected" : ""}`;
    chip.textContent = lens.name;
    chip.addEventListener("click", () => {
      setTopLevelState({ selectedLensId: lens.id }, "lens_selected");
      renderLensesStrip(lenses);
    });
    strip.appendChild(chip);
  }
}

function appendMessageCard(msg, autoPlay) {
  const feed = document.getElementById("feed");
  const card = document.createElement("div");
  card.className = `msg-card ${msg.role === "user" ? "user" : "assistant"}`;

  if (msg.role === "user") {
    card.innerHTML = `<div class="msg-body">${escapeHtml(msg.text)}</div>`;
  } else {
    const locTitle = msg.location?.short_label || "Current Location";
    const lensHtml = msg.lens?.name
      ? `<span class="lens-tag">${escapeHtml(msg.lens.name)}</span>`
      : `<span class="lens-tag">Follow-up</span>`;

    let sourcesHtml = "";
    if (msg.sources && msg.sources.length > 0) {
      const links = msg.sources
        .filter((s) => typeof s.uri === "string" && /^https?:\/\//i.test(s.uri))
        .map(
          (s) =>
            `<a class="source-link" href="${escapeHtml(s.uri)}" target="_blank" rel="noopener">${escapeHtml(s.title)}</a>`
        )
        .join("");
      sourcesHtml = `<div class="sources-list">${links}</div>`;
    }

    const playBtnHtml = msg.audio_url
      ? `<button class="play-clip-btn" data-audio-url="${escapeHtml(msg.audio_url)}" data-title="${escapeHtml(locTitle)}" data-subtitle="${escapeHtml(msg.lens?.name || "Hear")}">▶ Play Audio</button>`
      : `<button class="play-clip-btn" data-text="${escapeHtml(msg.text)}">🔊 Speak (Browser)</button>`;

    card.innerHTML = `
      <div class="msg-header">
        <span>📍 ${escapeHtml(locTitle)}</span>
        ${lensHtml}
      </div>
      <div class="msg-body">${escapeHtml(msg.text)}</div>
      <div class="msg-footer">
        ${sourcesHtml}
        ${playBtnHtml}
      </div>
    `;
  }

  feed.appendChild(card);
  feed.scrollTop = feed.scrollHeight;

  if (autoPlay && msg.role === "assistant") {
    if (msg.audio_url) {
      audio.playAudioUrl(
        msg.audio_url,
        msg.location?.short_label || "Hear",
        msg.lens?.name || "Historical Narration"
      );
    } else if (
      appState.currentConfig?.gemini?.tts_enabled === false &&
      "speechSynthesis" in window
    ) {
      audio.speakBrowserTts(msg.text);
    }
  }
  return card;
}

function addSystemBanner(text) {
  const feed = document.getElementById("feed");
  const div = document.createElement("div");
  div.className = "msg-card";
  div.style.borderColor = "var(--amber)";
  div.innerHTML = `<div class="msg-body" style="color: #fcd34d;">${escapeHtml(text)}</div>`;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}

// ---------------------------------------------------------------------------
// Pocket Mode (WakeLock + Pure-Black OLED Overlay)
// ---------------------------------------------------------------------------
let wakeLockRequestInFlight = false;

async function requestScreenWakeLock() {
  if (!("wakeLock" in navigator)) return;
  if (appState.device.wakeLockSentinel || wakeLockRequestInFlight) return;
  wakeLockRequestInFlight = true;
  try {
    const sentinel = await navigator.wakeLock.request("screen");
    if (!appState.device.pocketMode) {
      try {
        await sentinel.release();
      } catch (_e) {
        // Ignore already-released WakeLock errors
      }
      return;
    }
    sentinel.addEventListener("release", () => {
      if (appState.device.wakeLockSentinel === sentinel) {
        patchSection(
          "device",
          { wakeLock: WakeLockState.OFF, wakeLockSentinel: null },
          "wakelock_released"
        );
      }
    });
    patchSection(
      "device",
      { wakeLock: WakeLockState.ACTIVE, wakeLockSentinel: sentinel, wakeLockError: null },
      "wakelock_acquired"
    );
  } catch (e) {
    console.warn("WakeLock request failed:", e);
    patchSection(
      "device",
      { wakeLock: WakeLockState.ERROR, wakeLockError: e?.message || "wakelock_failed" },
      "wakelock_error"
    );
  } finally {
    wakeLockRequestInFlight = false;
  }
}

async function enterPocketMode() {
  audio.ensureSilentKeepAlive();
  document.getElementById("pocketOverlay").classList.add("active");
  patchSection("device", { pocketMode: true }, "pocket_enter");
  await requestScreenWakeLock();
}

async function exitPocketMode() {
  document.getElementById("pocketOverlay").classList.remove("active");
  patchSection("device", { pocketMode: false }, "pocket_exit");
  const sentinel = appState.device.wakeLockSentinel;
  if (sentinel) {
    try {
      await sentinel.release();
    } catch (_e) {
      // Ignore already-released WakeLock errors
    }
    if (appState.device.wakeLockSentinel === sentinel) {
      patchSection(
        "device",
        { wakeLock: WakeLockState.OFF, wakeLockSentinel: null },
        "wakelock_released"
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Voice Dictation (Web Speech API)
// ---------------------------------------------------------------------------
function toggleDictation() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    alert("Use the microphone icon on your Pixel keyboard to dictate.");
    document.getElementById("chatInput").focus();
    return;
  }
  const rec = new SpeechRec();
  rec.lang = "en-US";
  rec.interimResults = false;
  const micBtn = document.getElementById("micBtn");
  micBtn.classList.add("active");
  rec.onresult = (e) => {
    const transcript = e.results[0][0].transcript;
    document.getElementById("chatInput").value = transcript;
  };
  rec.onend = () => micBtn.classList.remove("active");
  rec.onerror = () => micBtn.classList.remove("active");
  rec.start();
}

init();
