import { PlaybackState, patchSection } from "./state.js";

export function createAudioController({
  mainAudio,
  audioBar,
  speedBtn,
  state,
  onAudioEnded,
  onNextTrack,
  onMediaSessionAction = () => {},
  onKeepAliveError = () => {},
}) {
  let keepAliveAudio = null;
  let activeSpeechUtterance = null;

  function setupAudioEvents() {
    mainAudio.addEventListener("play", () => {
      patchSection("playback", { status: PlaybackState.PLAYING, lastError: null }, "audio_play");
    });
    mainAudio.addEventListener("pause", () => {
      if (mainAudio.ended) return;
      patchSection(
        "playback",
        { status: mainAudio.currentTime > 0 ? PlaybackState.PAUSED : PlaybackState.IDLE },
        "audio_pause"
      );
    });
    mainAudio.addEventListener("ended", () => {
      patchSection("playback", { status: PlaybackState.IDLE }, "audio_ended");
      onAudioEnded();
    });
    mainAudio.addEventListener("error", () => {
      const code = mainAudio.error?.code;
      patchSection(
        "playback",
        {
          status: PlaybackState.ERROR,
          lastError: code ? `media_error_${code}` : "media_error",
        },
        "audio_error"
      );
    });
  }

  function playAudioUrl(url, title, subtitle) {
    audioBar.classList.add("visible");
    mainAudio.src = url;
    mainAudio.playbackRate = state.playback.speeds[state.playback.speedIdx];
    patchSection(
      "playback",
      { currentTitle: title || "Historical Narration", lastError: null },
      "audio_load"
    );
    mainAudio.play().catch((e) => {
      console.warn("Audio play blocked until user gesture:", e);
      patchSection(
        "playback",
        { status: PlaybackState.PAUSED, lastError: e?.name || "autoplay_blocked" },
        "audio_autoplay_blocked"
      );
    });

    if ("mediaSession" in navigator) {
      navigator.mediaSession.metadata = new MediaMetadata({
        title: title || "Historical Narration",
        artist: subtitle || "Hear Walking Companion",
        album: "Hear",
      });
    }
  }

  function speakBrowserTts(text) {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    activeSpeechUtterance = utter;
    utter.rate = state.playback.speeds[state.playback.speedIdx];
    utter.onstart = () => {
      if (activeSpeechUtterance === utter) {
        patchSection(
          "playback",
          { status: PlaybackState.PLAYING, currentTitle: "Browser TTS", lastError: null },
          "tts_play"
        );
      }
    };
    utter.onend = () => {
      if (activeSpeechUtterance !== utter) return;
      activeSpeechUtterance = null;
      patchSection("playback", { status: PlaybackState.IDLE }, "tts_ended");
      onAudioEnded();
    };
    utter.onerror = (ev) => {
      if (activeSpeechUtterance !== utter) return;
      activeSpeechUtterance = null;
      patchSection(
        "playback",
        { status: PlaybackState.ERROR, lastError: ev?.error || "tts_error" },
        "tts_error"
      );
    };
    window.speechSynthesis.speak(utter);
  }

  function cycleSpeed() {
    const nextIdx = (state.playback.speedIdx + 1) % state.playback.speeds.length;
    const rate = state.playback.speeds[nextIdx];
    mainAudio.playbackRate = rate;
    speedBtn.textContent = `${rate}x`;
    patchSection("playback", { speedIdx: nextIdx }, "audio_speed");
  }

  function setupMediaSession() {
    if (!("mediaSession" in navigator)) return;
    navigator.mediaSession.setActionHandler("play", () => {
      onMediaSessionAction("play");
      mainAudio.play();
    });
    navigator.mediaSession.setActionHandler("pause", () => {
      onMediaSessionAction("pause");
      mainAudio.pause();
    });
    // Double-tap on Pixel Buds ("nexttrack") triggers "Tell me about here"!
    navigator.mediaSession.setActionHandler("nexttrack", () => {
      onMediaSessionAction("nexttrack");
      onNextTrack("earbud_nexttrack");
    });
    navigator.mediaSession.setActionHandler("previoustrack", () => {
      onMediaSessionAction("previoustrack");
      if (mainAudio.src) {
        mainAudio.currentTime = 0;
        mainAudio.play();
      }
    });
  }

  // Plays an ultra-quiet generated WAV loop so Android Chrome keeps the MediaSession
  // and background tab alive even during silent walking intervals between narrations.
  function ensureSilentKeepAlive() {
    if (keepAliveAudio) return;
    try {
      const sampleRate = 8000;
      const numSamples = sampleRate; // 1 second
      const buffer = new ArrayBuffer(44 + numSamples * 2);
      const view = new DataView(buffer);
      const writeStr = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
      };
      writeStr(0, "RIFF");
      view.setUint32(4, 36 + numSamples * 2, true);
      writeStr(8, "WAVE");
      writeStr(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeStr(36, "data");
      view.setUint32(40, numSamples * 2, true);

      const blob = new Blob([buffer], { type: "audio/wav" });
      keepAliveAudio = new Audio(URL.createObjectURL(blob));
      keepAliveAudio.loop = true;
      keepAliveAudio.volume = 0.01;
      keepAliveAudio
        .play()
        .then(() => {
          patchSection("playback", { keepAliveActive: true }, "keepalive_started");
        })
        .catch((err) => {
          onKeepAliveError(err?.name || err?.message || "play_rejected");
        });
    } catch (e) {
      console.warn("Silent keep-alive init failed:", e);
      onKeepAliveError(e?.message || "init_failed");
    }
  }

  function stopAndReset() {
    mainAudio.pause();
    mainAudio.removeAttribute("src");
    mainAudio.load();
    audioBar.classList.remove("visible");
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    activeSpeechUtterance = null;
    patchSection(
      "playback",
      { status: PlaybackState.IDLE, currentTitle: null, lastError: null },
      "audio_reset"
    );
  }

  return {
    setupAudioEvents,
    setupMediaSession,
    playAudioUrl,
    speakBrowserTts,
    cycleSpeed,
    ensureSilentKeepAlive,
    stopAndReset,
  };
}
