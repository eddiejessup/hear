import { apiFetch } from "./api.js";
import { createAudioController } from "./audio.js";
import {
  createDrawersController,
  renderDebugInfo,
  updateAutoTriggerButton,
} from "./drawers.js";
import { appState, approxDistanceMeters, escapeHtml } from "./state.js";

const mainAudio = document.getElementById("mainAudio");
const audioBar = document.getElementById("audioBar");
const speedBtn = document.getElementById("speedBtn");

const audio = createAudioController({
  mainAudio,
  audioBar,
  speedBtn,
  state: appState,
  onAudioEnded: () => sendPing(true),
  onNextTrack: (reason) => triggerNarration(reason),
});

const drawers = createDrawersController({
  state: appState,
  audio,
  renderLensesStrip,
  sendPing,
  startGpsWatch,
  updateLocationHeader,
});

// ---------------------------------------------------------------------------
// Initialization & Event Wiring
// ---------------------------------------------------------------------------
function bindDomEvents() {
  document.getElementById("versionBadge").addEventListener("click", drawers.openSimDrawer);
  document.getElementById("autoToggleBtn").addEventListener("click", drawers.toggleAutoTrigger);
  document.getElementById("pocketBtn").addEventListener("click", enterPocketMode);
  document.getElementById("openSimBtn").addEventListener("click", drawers.openSimDrawer);
  document.getElementById("openConfigBtn").addEventListener("click", drawers.openConfigDrawer);

  document
    .getElementById("narrateBtn")
    .addEventListener("click", () => triggerNarration("manual"));
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
  document.getElementById("nudgeNorthBtn").addEventListener("click", () => drawers.nudgeSim(150, 0));
  document.getElementById("nudgeEastBtn").addEventListener("click", () => drawers.nudgeSim(0, 150));
  document.getElementById("resumeGpsBtn").addEventListener("click", drawers.resumeRealGps);
  document.getElementById("resetSessionBtn").addEventListener("click", drawers.resetSession);

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
}

async function init() {
  bindDomEvents();
  audio.setupAudioEvents();
  audio.setupMediaSession();
  await fetchInitialState();
  startGpsWatch();
  setInterval(periodicPing, 6000);
}

async function fetchInitialState() {
  try {
    const res = await apiFetch("/api/state");
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      addSystemBanner(`⚠️ ${err.detail || res.statusText}`);
      return;
    }
    const data = await res.json();
    appState.currentConfig = data.config;
    renderLensesStrip(appState.currentConfig.lenses || []);
    updateAutoTriggerButton(appState.currentConfig.triggers?.auto_trigger_enabled ?? true);
    renderDebugInfo(data.debug_info, data.session_id);

    if (!data.has_api_key) {
      addSystemBanner(
        "⚠️ GEMINI_API_KEY is not set on the server. Set GEMINI_API_KEY in your environment or .env file."
      );
    }

    if (data.current_location) {
      appState.currentCoords = data.current_location;
      updateLocationHeader(data.current_location, data.trigger_status);
    }

    if (data.messages && data.messages.length > 0) {
      document.getElementById("emptyState").style.display = "none";
      for (const msg of data.messages) {
        appendMessageCard(msg, false);
      }
    }
  } catch (err) {
    console.error("Failed to load initial state:", err);
  }
}

// ---------------------------------------------------------------------------
// GPS Tracking & Server Ping Loop
// ---------------------------------------------------------------------------
function startGpsWatch() {
  if (!navigator.geolocation) {
    document.getElementById("locLabel").textContent = "Geolocation not supported in this browser";
    document.getElementById("gpsDot").className = "gps-dot error";
    return;
  }
  if (appState.watchId !== null) {
    navigator.geolocation.clearWatch(appState.watchId);
  }
  appState.watchId = navigator.geolocation.watchPosition(
    (pos) => {
      if (appState.useSimulatedGps) return;
      document.getElementById("gpsDot").className = "gps-dot active";
      appState.currentCoords = {
        lat: pos.coords.latitude,
        lon: pos.coords.longitude,
        accuracy: pos.coords.accuracy,
        heading: pos.coords.heading,
        speed: pos.coords.speed,
      };
      sendPing(false);
    },
    (err) => {
      if (appState.useSimulatedGps) return;
      console.warn("GPS watch error:", err.message);
      document.getElementById("gpsDot").className = "gps-dot error";
      document.getElementById("locLabel").textContent = `GPS waiting (${err.message}) — or use Sim`;
    },
    {
      enableHighAccuracy: true,
      maximumAge: 5000,
      timeout: 15000,
    }
  );
}

async function periodicPing() {
  if (!appState.currentCoords || appState.isGenerating) return;
  const autoEnabled = appState.currentConfig?.triggers?.auto_trigger_enabled ?? false;
  if (
    !autoEnabled &&
    appState.lastPingSnapshot &&
    appState.lastPingSnapshot.isPlayingAudio === appState.isPlayingAudio &&
    appState.lastPingSnapshot.isPausedAudio === appState.isPausedAudio &&
    approxDistanceMeters(
      appState.currentCoords.lat,
      appState.currentCoords.lon,
      appState.lastPingSnapshot.lat,
      appState.lastPingSnapshot.lon
    ) < 3.0
  ) {
    return;
  }
  await sendPing(false);
}

async function sendPing(audioJustEnded = false) {
  if (!appState.currentCoords) return;
  if (appState.pingInFlight && !audioJustEnded) return;
  appState.pingInFlight = true;
  try {
    const res = await apiFetch("/api/ping", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lat: appState.currentCoords.lat,
        lon: appState.currentCoords.lon,
        accuracy: appState.currentCoords.accuracy ?? null,
        heading: appState.currentCoords.heading ?? null,
        speed: appState.currentCoords.speed ?? null,
        is_playing_audio: appState.isPlayingAudio,
        is_paused_audio: appState.isPausedAudio,
        audio_just_ended: audioJustEnded,
      }),
    });
    if (!res.ok) return;
    const data = await res.json();
    if (data.location?.short_label && appState.currentCoords) {
      appState.currentCoords = {
        ...appState.currentCoords,
        short_label: data.location.short_label,
      };
    }
    appState.lastPingSnapshot = {
      lat: appState.currentCoords.lat,
      lon: appState.currentCoords.lon,
      isPlayingAudio: appState.isPlayingAudio,
      isPausedAudio: appState.isPausedAudio,
    };
    updateLocationHeader(data.location, data.trigger_status);

    if (
      data.trigger_status?.should_trigger &&
      !appState.isGenerating &&
      !appState.isPlayingAudio
    ) {
      triggerNarration(data.trigger_status.trigger_reason || "auto");
    }
  } catch (err) {
    console.warn("Ping failed:", err);
  } finally {
    appState.pingInFlight = false;
  }
}

function updateLocationHeader(loc, status) {
  if (loc) {
    document.getElementById("locLabel").textContent =
      loc.short_label || `${loc.lat.toFixed(5)}, ${loc.lon.toFixed(5)}`;
    const acc = loc.accuracy
      ? ` ±${Math.round(loc.accuracy)}m`
      : appState.useSimulatedGps
        ? " (sim)"
        : "";
    document.getElementById("coordBadge").textContent =
      `${loc.lat.toFixed(5)}, ${loc.lon.toFixed(5)}${acc}`;
  }
  if (status) {
    const dist =
      status.distance_moved_m !== null && status.distance_moved_m !== undefined
        ? `${Math.round(status.distance_moved_m)}m`
        : "0m";
    document.getElementById("distBadge").textContent =
      `Moved: ${dist} / ${Math.round(status.min_distance_m)}m`;

    const cdBadge = document.getElementById("cooldownBadge");
    if (status.is_generating) {
      cdBadge.textContent = "Researching…";
    } else if (appState.isPlayingAudio) {
      cdBadge.textContent = "Playing audio";
    } else if (status.is_paused_audio) {
      cdBadge.textContent = "Paused (holding)";
    } else if (status.cooldown_remaining_s > 0) {
      cdBadge.textContent = `Cooldown: ${Math.ceil(status.cooldown_remaining_s)}s`;
    } else {
      cdBadge.textContent = status.auto_trigger_enabled ? "Armed" : "Manual";
    }

    const pocketStatus = document.getElementById("pocketStatus");
    if (pocketStatus) {
      pocketStatus.textContent = `${loc?.short_label || ""} • Moved ${dist}`;
    }
  }
}

// ---------------------------------------------------------------------------
// Narration & Chat Steering
// ---------------------------------------------------------------------------
async function triggerNarration(reason = "manual") {
  if (appState.isGenerating) return;
  if (!appState.currentCoords) {
    alert(
      "Waiting for GPS coordinates (or open 'Sim' in the top bar to test with simulated coordinates)."
    );
    return;
  }

  audio.ensureSilentKeepAlive();
  appState.isGenerating = true;
  const btn = document.getElementById("narrateBtn");
  btn.disabled = true;
  btn.innerHTML = `<span>⏳ Researching &amp; generating audio…</span>`;
  document.getElementById("cooldownBadge").textContent = "Researching…";

  try {
    const res = await apiFetch("/api/narrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lat: appState.currentCoords.lat,
        lon: appState.currentCoords.lon,
        heading: appState.currentCoords.heading ?? null,
        lens_id: appState.selectedLensId || null,
        reason,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      if (res.status !== 409) {
        addSystemBanner(`Error: ${data.detail || res.statusText}`);
      }
      return;
    }

    const emptyEl = document.getElementById("emptyState");
    if (emptyEl) emptyEl.style.display = "none";
    if (data.message?.location?.short_label && appState.currentCoords) {
      appState.currentCoords = {
        ...appState.currentCoords,
        short_label: data.message.location.short_label,
      };
    }
    appendMessageCard(data.message, true);
    updateLocationHeader(appState.currentCoords, data.trigger_status);
  } catch (err) {
    addSystemBanner(`Network error: ${err.message}`);
  } finally {
    appState.isGenerating = false;
    btn.disabled = false;
    btn.innerHTML = `<span>🎙️ Tell me about here</span>`;
  }
}

async function sendChat(event) {
  event.preventDefault();
  const input = document.getElementById("chatInput");
  const text = input.value.trim();
  if (!text || appState.isGenerating) return;

  audio.ensureSilentKeepAlive();
  input.value = "";
  appState.isGenerating = true;
  document.getElementById("sendBtn").disabled = true;

  const emptyEl = document.getElementById("emptyState");
  if (emptyEl) emptyEl.style.display = "none";
  const userCard = appendMessageCard(
    {
      role: "user",
      text,
      location_label: appState.currentCoords?.short_label,
    },
    false
  );

  try {
    const res = await apiFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        lat: appState.currentCoords?.lat ?? null,
        lon: appState.currentCoords?.lon ?? null,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      userCard?.remove();
      input.value = text;
      addSystemBanner(`Error: ${data.detail || res.statusText}`);
      return;
    }
    if (data.assistant_message?.location?.short_label && appState.currentCoords) {
      appState.currentCoords = {
        ...appState.currentCoords,
        short_label: data.assistant_message.location.short_label,
      };
    }
    appendMessageCard(data.assistant_message, true);
    updateLocationHeader(appState.currentCoords, data.trigger_status);
  } catch (err) {
    userCard?.remove();
    input.value = text;
    addSystemBanner(`Chat error: ${err.message}`);
  } finally {
    appState.isGenerating = false;
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
    appState.selectedLensId = "";
    renderLensesStrip(lenses);
  });
  strip.appendChild(randomChip);

  for (const lens of lenses) {
    if (lens.enabled === false) continue;
    const chip = document.createElement("button");
    chip.className = `lens-chip ${appState.selectedLensId === lens.id ? "selected" : ""}`;
    chip.textContent = lens.name;
    chip.addEventListener("click", () => {
      appState.selectedLensId = lens.id;
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
async function enterPocketMode() {
  audio.ensureSilentKeepAlive();
  document.getElementById("pocketOverlay").classList.add("active");
  if ("wakeLock" in navigator) {
    try {
      appState.wakeLockSentinel = await navigator.wakeLock.request("screen");
    } catch (e) {
      console.warn("WakeLock request failed:", e);
    }
  }
}

async function exitPocketMode() {
  document.getElementById("pocketOverlay").classList.remove("active");
  if (appState.wakeLockSentinel) {
    try {
      await appState.wakeLockSentinel.release();
    } catch (_e) {
      // Ignore already-released WakeLock errors
    }
    appState.wakeLockSentinel = null;
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
