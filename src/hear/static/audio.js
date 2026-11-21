export function createAudioController({
  mainAudio,
  audioBar,
  speedBtn,
  state,
  onAudioEnded,
  onNextTrack,
}) {
  let keepAliveAudio = null;
  let activeSpeechUtterance = null;

  function setupAudioEvents() {
    mainAudio.addEventListener("play", () => {
      state.isPlayingAudio = true;
      state.isPausedAudio = false;
    });
    mainAudio.addEventListener("pause", () => {
      if (mainAudio.ended) return;
      state.isPlayingAudio = false;
      state.isPausedAudio = mainAudio.currentTime > 0;
    });
    mainAudio.addEventListener("ended", () => {
      state.isPlayingAudio = false;
      state.isPausedAudio = false;
      onAudioEnded();
    });
    mainAudio.addEventListener("error", () => {
      state.isPlayingAudio = false;
      state.isPausedAudio = false;
    });
  }

  function playAudioUrl(url, title, subtitle) {
    audioBar.classList.add("visible");
    mainAudio.src = url;
    mainAudio.playbackRate = state.playbackSpeeds[state.speedIdx];
    mainAudio.play().catch((e) => console.warn("Audio play blocked until user gesture:", e));

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
    utter.rate = state.playbackSpeeds[state.speedIdx];
    utter.onstart = () => {
      if (activeSpeechUtterance === utter) state.isPlayingAudio = true;
    };
    utter.onend = () => {
      if (activeSpeechUtterance !== utter) return;
      activeSpeechUtterance = null;
      state.isPlayingAudio = false;
      onAudioEnded();
    };
    utter.onerror = () => {
      if (activeSpeechUtterance !== utter) return;
      activeSpeechUtterance = null;
      state.isPlayingAudio = false;
    };
    window.speechSynthesis.speak(utter);
  }

  function cycleSpeed() {
    state.speedIdx = (state.speedIdx + 1) % state.playbackSpeeds.length;
    const rate = state.playbackSpeeds[state.speedIdx];
    mainAudio.playbackRate = rate;
    speedBtn.textContent = `${rate}x`;
  }

  function setupMediaSession() {
    if (!("mediaSession" in navigator)) return;
    navigator.mediaSession.setActionHandler("play", () => mainAudio.play());
    navigator.mediaSession.setActionHandler("pause", () => mainAudio.pause());
    // Double-tap on Pixel Buds ("nexttrack") triggers "Tell me about here"!
    navigator.mediaSession.setActionHandler("nexttrack", () => {
      onNextTrack("earbud_nexttrack");
    });
    navigator.mediaSession.setActionHandler("previoustrack", () => {
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
      keepAliveAudio.play().catch(() => {});
    } catch (e) {
      console.warn("Silent keep-alive init failed:", e);
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
    state.isPlayingAudio = false;
    state.isPausedAudio = false;
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
