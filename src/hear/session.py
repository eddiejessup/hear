import asyncio
import random
import time
from datetime import UTC, datetime

from hear.geo import haversine_meters
from hear.models import (
    AppConfig,
    CurrentLocation,
    GeminiContent,
    LensConfig,
    SessionTranscript,
    TriggerLocation,
    TriggerStatus,
    UIMessage,
)
from hear.paths import SESSIONS_DIR


class WalkSession:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.session_id: str = ""
        self.started_at: float = 0.0
        self.gemini_contents: list[GeminiContent] = []
        self.ui_messages: list[UIMessage] = []
        self.used_lens_ids: list[str] = []
        self.current_location: CurrentLocation | None = None
        self.last_trigger_location: TriggerLocation | None = None
        self.last_trigger_time: float | None = None
        self.last_audio_ended_at: float | None = None
        self.is_generating: bool = False
        self.is_playing_audio: bool = False
        self.reset()

    def reset(self) -> None:
        self.session_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.started_at = time.time()
        self.gemini_contents = []
        self.ui_messages = []
        self.used_lens_ids = []
        self.current_location = None
        self.last_trigger_location = None
        self.last_trigger_time = None
        self.last_audio_ended_at = None
        self.is_generating = False
        self.is_playing_audio = False

    def sample_lens(self, config: AppConfig, requested_lens_id: str | None = None) -> LensConfig:
        all_lenses = config.lenses
        enabled_lenses = [lens for lens in all_lenses if lens.enabled]
        if not enabled_lenses:
            enabled_lenses = all_lenses or [
                LensConfig(
                    id="general_history",
                    name="Deep Local History",
                    prompt=(
                        "Explore the structural, economic, and political history "
                        "of this immediate location."
                    ),
                )
            ]

        if requested_lens_id:
            for lens in all_lenses:
                if lens.id == requested_lens_id:
                    if lens.id and lens.id not in self.used_lens_ids:
                        self.used_lens_ids.append(lens.id)
                    return lens

        available = [lens for lens in enabled_lenses if lens.id not in self.used_lens_ids]
        if not available:
            self.used_lens_ids.clear()
            available = enabled_lenses

        chosen = random.choice(available)
        if chosen.id:
            self.used_lens_ids.append(chosen.id)
        return chosen

    def compute_trigger_status(self, config: AppConfig) -> TriggerStatus:
        trig = config.triggers
        auto_enabled = trig.auto_trigger_enabled
        min_dist_m = trig.min_distance_meters
        min_cooldown_s = trig.min_cooldown_seconds
        max_silence_s = trig.max_silence_seconds
        min_drift_m = trig.min_drift_for_silence_trigger_meters

        now = time.time()
        dist_moved_m: float | None = None
        if self.current_location and self.last_trigger_location:
            dist_moved_m = haversine_meters(
                self.current_location.lat,
                self.current_location.lon,
                self.last_trigger_location.lat,
                self.last_trigger_location.lon,
            )

        ref_time = self.last_audio_ended_at or self.last_trigger_time
        silence_elapsed_s = (now - ref_time) if ref_time else None
        cooldown_remaining_s = 0.0
        if silence_elapsed_s is not None:
            cooldown_remaining_s = max(0.0, min_cooldown_s - silence_elapsed_s)

        should_trigger = False
        trigger_reason: str | None = None

        if (
            auto_enabled
            and not self.is_generating
            and not self.is_playing_audio
            and self.current_location
        ):
            if self.last_trigger_location is None:
                should_trigger = True
                trigger_reason = "initial_location"
            elif cooldown_remaining_s <= 0:
                if dist_moved_m is not None and dist_moved_m >= min_dist_m:
                    should_trigger = True
                    trigger_reason = f"distance ({int(dist_moved_m)}m >= {int(min_dist_m)}m)"
                elif (
                    silence_elapsed_s is not None
                    and silence_elapsed_s >= max_silence_s
                    and (dist_moved_m is None or dist_moved_m >= min_drift_m)
                ):
                    should_trigger = True
                    trigger_reason = (
                        f"max_silence ({int(silence_elapsed_s)}s >= {int(max_silence_s)}s)"
                    )

        return TriggerStatus(
            auto_trigger_enabled=auto_enabled,
            should_trigger=should_trigger,
            trigger_reason=trigger_reason,
            distance_moved_m=round(dist_moved_m, 1) if dist_moved_m is not None else None,
            min_distance_m=min_dist_m,
            silence_elapsed_s=(
                round(silence_elapsed_s, 1) if silence_elapsed_s is not None else None
            ),
            cooldown_remaining_s=round(cooldown_remaining_s, 1),
            min_cooldown_s=min_cooldown_s,
            max_silence_s=max_silence_s,
            is_generating=self.is_generating,
            is_playing_audio=self.is_playing_audio,
        )

    def persist_transcript(self) -> None:
        out_file = SESSIONS_DIR / f"session_{self.session_id}.json"
        transcript = SessionTranscript(
            session_id=self.session_id,
            started_at=self.started_at,
            messages=self.ui_messages,
        )
        out_file.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")


session = WalkSession()
