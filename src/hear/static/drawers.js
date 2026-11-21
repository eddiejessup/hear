import { apiFetch } from "./api.js";
import {
  escapeHtml,
  formatUptime,
  GpsState,
  parseNonNegativeFloat,
  patchSection,
  setTopLevelState,
} from "./state.js";

function setSelectValuePreservingCustom(selectEl, value, fallback) {
  const resolved = value || fallback;
  if (resolved && !Array.from(selectEl.options).some((opt) => opt.value === resolved)) {
    const customOpt = document.createElement("option");
    customOpt.value = resolved;
    customOpt.textContent = `${resolved} (Custom)`;
    selectEl.appendChild(customOpt);
  }
  selectEl.value = resolved;
}

export function renderDebugInfo(info, sessionId) {
  if (!info) return;
  const badge = document.getElementById("versionBadge");
  if (badge) {
    const revShort =
      info.cloud_run_revision && info.cloud_run_revision !== "local"
        ? ` • ${info.cloud_run_revision.replace(/^hear-/, "")}`
        : " • local";
    badge.textContent = `${info.git_commit}${revShort}`;
  }
  const box = document.getElementById("debugInfoBox");
  if (box) {
    box.innerHTML = [
      `<div><strong>Git Commit:</strong> ${escapeHtml(info.git_commit)} (${escapeHtml(info.git_commit_time)})</div>`,
      `<div><strong>Built At:</strong> ${escapeHtml(info.build_time)}</div>`,
      `<div><strong>Revision:</strong> ${escapeHtml(info.cloud_run_revision)} (${escapeHtml(info.cloud_run_service)})</div>`,
      `<div><strong>Server Boot:</strong> ${escapeHtml(info.server_started_at)} (up ${formatUptime(info.server_uptime_seconds)})</div>`,
      `<div><strong>Python:</strong> ${escapeHtml(info.python_version)}</div>`,
      `<div><strong>Session ID:</strong> ${escapeHtml(sessionId || "--")}</div>`,
    ].join("");
  }
}

export function updateAutoTriggerButton(enabled) {
  const btn = document.getElementById("autoToggleBtn");
  btn.textContent = enabled ? "Auto: ON" : "Auto: OFF";
  btn.className = enabled ? "icon-btn active" : "icon-btn";
}

export function createDrawersController({
  state,
  audio,
  renderLensesStrip,
  sendPing,
  startGpsWatch,
}) {
  async function refreshDebugInfoOnly() {
    try {
      const res = await apiFetch("/api/state");
      if (!res.ok) return;
      const data = await res.json();
      setTopLevelState({ sessionId: data.session_id }, "session_refresh");
      renderDebugInfo(data.debug_info, data.session_id);
    } catch (_e) {
      // Best-effort refresh when opening simulator drawer
    }
  }

  function openSimDrawer() {
    refreshDebugInfoOnly();
    if (state.gps.coords) {
      document.getElementById("simLat").value = state.gps.coords.lat.toFixed(5);
      document.getElementById("simLon").value = state.gps.coords.lon.toFixed(5);
    }
    document.getElementById("simBackdrop").classList.add("open");
  }

  function closeSimDrawer() {
    document.getElementById("simBackdrop").classList.remove("open");
  }

  function applySimCoords() {
    const lat = parseFloat(document.getElementById("simLat").value);
    const lon = parseFloat(document.getElementById("simLon").value);
    if (Number.isNaN(lat) || Number.isNaN(lon)) return;
    patchSection(
      "gps",
      {
        simulated: true,
        status: GpsState.SIMULATED,
        coords: { lat, lon, accuracy: 5 },
        lastFixAt: Date.now(),
        lastError: null,
      },
      "gps_simulated"
    );
    sendPing(false);
    closeSimDrawer();
  }

  function setSimPreset(lat, lon) {
    document.getElementById("simLat").value = lat;
    document.getElementById("simLon").value = lon;
    applySimCoords();
  }

  function nudgeSim(northMeters, eastMeters) {
    const baseLat = state.gps.coords ? state.gps.coords.lat : 51.5142;
    const baseLon = state.gps.coords ? state.gps.coords.lon : -0.0885;
    const dLat = northMeters / 111320;
    const dLon = eastMeters / (111320 * Math.cos((baseLat * Math.PI) / 180));
    document.getElementById("simLat").value = (baseLat + dLat).toFixed(5);
    document.getElementById("simLon").value = (baseLon + dLon).toFixed(5);
    applySimCoords();
  }

  function resumeRealGps() {
    patchSection(
      "gps",
      { simulated: false, status: GpsState.WAITING, lastError: null },
      "gps_real"
    );
    closeSimDrawer();
    startGpsWatch();
  }

  async function resetSession() {
    const res = await apiFetch("/api/session/reset", { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      audio.stopAndReset();
      setTopLevelState({ sessionId: data.session_id }, "session_reset");
      patchSection("network", { lastPingSnapshot: null }, "session_reset");
      patchSection(
        "trigger",
        { status: data.trigger_status, updatedAt: Date.now() },
        "session_reset"
      );
      document.getElementById("feed").innerHTML =
        `<div id="emptyState" class="empty-state">New walk session started.</div>`;
      closeSimDrawer();
    }
  }

  async function toggleAutoTrigger() {
    try {
      const cfgRes = await apiFetch("/api/config");
      if (cfgRes.ok) {
        setTopLevelState({ currentConfig: await cfgRes.json() }, "config_loaded");
      }
    } catch (_e) {
      // Fall back to cached config
    }
    if (!state.currentConfig) return;
    state.currentConfig.triggers = state.currentConfig.triggers || {};
    state.currentConfig.triggers.auto_trigger_enabled =
      !state.currentConfig.triggers.auto_trigger_enabled;
    updateAutoTriggerButton(state.currentConfig.triggers.auto_trigger_enabled);
    const res = await apiFetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.currentConfig),
    });
    if (res.ok) {
      const data = await res.json();
      setTopLevelState({ currentConfig: data.config }, "config_updated");
      patchSection(
        "trigger",
        { status: data.trigger_status, updatedAt: Date.now() },
        "config_updated"
      );
      renderLensesStrip(state.currentConfig.lenses || []);
    }
    sendPing(false);
  }

  function createLensRowElement({ id, name, enabled, prompt }) {
    const div = document.createElement("div");
    div.className = "lens-editor-item";
    div.innerHTML = `
      <div style="display:flex; gap:8px; align-items:center;">
        <input type="checkbox" class="lens-enabled" ${enabled ? "checked" : ""} />
        <input type="text" class="lens-name" value="${escapeHtml(name)}" placeholder="Lens Name" style="flex:1;" />
        <button type="button" class="icon-btn lens-remove-btn">🗑️</button>
      </div>
      <input type="hidden" class="lens-id" value="${escapeHtml(id)}" />
      <textarea class="lens-prompt" style="min-height:54px;" placeholder="Prompt instructions for this lens…">${escapeHtml(prompt)}</textarea>
    `;
    return div;
  }

  function renderConfigLenses(lenses) {
    const container = document.getElementById("cfgLensesList");
    container.innerHTML = "";
    lenses.forEach((lens, idx) => {
      container.appendChild(
        createLensRowElement({
          id: lens.id || `lens_${idx}`,
          name: lens.name || "",
          enabled: lens.enabled !== false,
          prompt: lens.prompt || "",
        })
      );
    });
  }

  function addLensRow() {
    const container = document.getElementById("cfgLensesList");
    const id = `custom_${Math.random().toString(36).slice(2, 7)}`;
    container.appendChild(
      createLensRowElement({
        id,
        name: "",
        enabled: true,
        prompt: "",
      })
    );
  }

  async function openConfigDrawer() {
    try {
      const cfgRes = await apiFetch("/api/config");
      if (cfgRes.ok) {
        setTopLevelState({ currentConfig: await cfgRes.json() }, "config_loaded");
      }
    } catch (_e) {
      // Fall back to cached config
    }
    if (!state.currentConfig) return;
    const g = state.currentConfig.gemini || {};
    const t = state.currentConfig.triggers || {};

    setSelectValuePreservingCustom(
      document.getElementById("cfgModel"),
      g.model,
      "gemini-4-hydrogen"
    );
    document.getElementById("cfgThinkingLevel").value = g.thinking_level || "";
    setSelectValuePreservingCustom(document.getElementById("cfgVoice"), g.tts_voice, "Charon");
    document.getElementById("cfgTtsEnabled").value = String(g.tts_enabled ?? true);
    document.getElementById("cfgTtsStyle").value = g.tts_style_prompt || "";
    document.getElementById("cfgMinDist").value = t.min_distance_meters ?? 120;
    document.getElementById("cfgMinCooldown").value = t.min_cooldown_seconds ?? 90;
    document.getElementById("cfgMaxSilence").value = t.max_silence_seconds ?? 300;
    document.getElementById("cfgSystemPrompt").value = state.currentConfig.system_prompt || "";

    renderConfigLenses(state.currentConfig.lenses || []);
    document.getElementById("configBackdrop").classList.add("open");
  }

  function closeConfigDrawer() {
    document.getElementById("configBackdrop").classList.remove("open");
  }

  async function saveConfigToServer() {
    const updated = JSON.parse(JSON.stringify(state.currentConfig || {}));
    updated.gemini = updated.gemini || {};
    updated.gemini.model =
      document.getElementById("cfgModel").value || updated.gemini.model || "gemini-4-hydrogen";
    updated.gemini.thinking_level = document.getElementById("cfgThinkingLevel").value || null;
    updated.gemini.tts_voice =
      document.getElementById("cfgVoice").value || updated.gemini.tts_voice || "Charon";
    updated.gemini.tts_enabled = document.getElementById("cfgTtsEnabled").value === "true";
    updated.gemini.tts_style_prompt = document.getElementById("cfgTtsStyle").value;

    updated.triggers = updated.triggers || {};
    updated.triggers.min_distance_meters = parseNonNegativeFloat(
      document.getElementById("cfgMinDist").value,
      120
    );
    updated.triggers.min_cooldown_seconds = parseNonNegativeFloat(
      document.getElementById("cfgMinCooldown").value,
      90
    );
    updated.triggers.max_silence_seconds = parseNonNegativeFloat(
      document.getElementById("cfgMaxSilence").value,
      300
    );

    updated.system_prompt = document.getElementById("cfgSystemPrompt").value;

    const lensItems = document.querySelectorAll("#cfgLensesList .lens-editor-item");
    const newLenses = [];
    lensItems.forEach((item) => {
      const name = item.querySelector(".lens-name").value.trim();
      const prompt = item.querySelector(".lens-prompt").value.trim();
      const id =
        item.querySelector(".lens-id").value.trim() ||
        name.toLowerCase().replace(/[^a-z0-9]+/g, "_");
      const enabled = item.querySelector(".lens-enabled").checked;
      if (name && prompt) {
        newLenses.push({ id, name, enabled, prompt });
      }
    });
    updated.lenses = newLenses;

    const res = await apiFetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updated),
    });
    if (res.ok) {
      const data = await res.json();
      setTopLevelState({ currentConfig: data.config }, "config_saved");
      patchSection(
        "trigger",
        { status: data.trigger_status, updatedAt: Date.now() },
        "config_saved"
      );
      renderLensesStrip(state.currentConfig.lenses || []);
      closeConfigDrawer();
      sendPing(false);
    }
  }

  return {
    openSimDrawer,
    closeSimDrawer,
    setSimPreset,
    applySimCoords,
    nudgeSim,
    resumeRealGps,
    resetSession,
    toggleAutoTrigger,
    openConfigDrawer,
    closeConfigDrawer,
    addLensRow,
    saveConfigToServer,
  };
}
