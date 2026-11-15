import asyncio
import random
import re
import time
import uuid
from datetime import UTC, datetime

from hear.gemini import call_gemini_text, call_gemini_tts
from hear.geo import compass_bearing_label, haversine_meters
from hear.models.config import AppConfig, LensConfig
from hear.models.gemini import GeminiContent, GeminiPart, GeminiTextResult
from hear.models.geo import GeoContext
from hear.models.session import (
    AssistantMessage,
    CurrentLocation,
    LensSummary,
    MessageLocation,
    SessionTranscript,
    TriggerLocation,
    TriggerStatus,
    UIMessage,
    UserChatMessage,
)
from hear.paths import SESSIONS_DIR

MAX_ACTIVE_SESSIONS = 64
PAUSE_HOLD_MAX_SECONDS = 180.0
PAUSE_HOLD_MAX_DISTANCE_METERS = 150.0
INITIAL_LOCATION_CHAT_GRACE_SECONDS = 15.0
_SAFE_CLIENT_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def sanitize_client_id(raw_client_id: str | None) -> str:
    if not raw_client_id:
        return "default"
    cleaned = _SAFE_CLIENT_ID_RE.sub("", raw_client_id.strip())[:48]
    return cleaned or "default"


class WalkSession:
    def __init__(self, client_id: str = "default") -> None:
        self.client_id: str = sanitize_client_id(client_id)
        self.lock = asyncio.Lock()
        self.session_id: str = ""
        self.started_at: float = 0.0
        self.last_active_at: float = time.time()
        self.gemini_contents: list[GeminiContent] = []
        self.ui_messages: list[UIMessage] = []
        self.used_lens_ids: list[str] = []
        self.current_location: CurrentLocation | None = None
        self.last_trigger_location: TriggerLocation | None = None
        self.last_trigger_time: float | None = None
        self.last_audio_ended_at: float | None = None
        self.is_generating: bool = False
        self.is_playing_audio: bool = False
        self.is_paused_audio: bool = False
        self.paused_at: float | None = None
        self.paused_location: tuple[float, float] | None = None
        self.reset()

    def reset(self) -> None:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        suffix = self.client_id[:8]
        self.session_id = f"{ts}_{suffix}" if suffix != "default" else ts
        self.started_at = time.time()
        self.last_active_at = self.started_at
        self.gemini_contents = []
        self.ui_messages = []
        self.used_lens_ids = []
        self.current_location = None
        self.last_trigger_location = None
        self.last_trigger_time = None
        self.last_audio_ended_at = None
        self.is_generating = False
        self.is_playing_audio = False
        self.is_paused_audio = False
        self.paused_at = None
        self.paused_location = None

    def update_audio_state(
        self,
        *,
        lat: float,
        lon: float,
        is_playing_audio: bool,
        is_paused_audio: bool = False,
        audio_just_ended: bool = False,
        now: float | None = None,
    ) -> None:
        ts = now if now is not None else time.time()
        if audio_just_ended or (
            self.is_playing_audio and not is_playing_audio and not is_paused_audio
        ):
            self.last_audio_ended_at = ts

        if is_paused_audio and not is_playing_audio:
            if not self.is_paused_audio or self.paused_at is None:
                self.paused_at = ts
                self.paused_location = (lat, lon)
        else:
            self.paused_at = None
            self.paused_location = None

        self.is_playing_audio = is_playing_audio
        self.is_paused_audio = is_paused_audio and not is_playing_audio

    def is_pause_hold_active(self, now: float) -> bool:
        """Return True if audio is paused mid-clip and hasn't expired via time + distance."""
        if not self.is_paused_audio:
            return False
        if self.paused_at is None or self.paused_location is None or self.current_location is None:
            return True
        paused_elapsed_s = now - self.paused_at
        dist_from_pause_m = haversine_meters(
            self.current_location.lat,
            self.current_location.lon,
            self.paused_location[0],
            self.paused_location[1],
        )
        return not (
            paused_elapsed_s >= PAUSE_HOLD_MAX_SECONDS
            and dist_from_pause_m >= PAUSE_HOLD_MAX_DISTANCE_METERS
        )

    def resolve_coords(
        self,
        lat: float | None,
        lon: float | None,
        heading: float | None = None,
    ) -> tuple[float | None, float | None, float | None]:
        cur = self.current_location
        resolved_lat = lat if lat is not None else (cur.lat if cur else None)
        resolved_lon = lon if lon is not None else (cur.lon if cur else None)
        resolved_heading = heading if heading is not None else (cur.heading if cur else None)
        return resolved_lat, resolved_lon, resolved_heading

    async def run_gemini_turn(
        self,
        config: AppConfig,
        user_prompt_text: str,
    ) -> tuple[GeminiTextResult, str | None, float]:
        """Run a Gemini text + optional TTS turn and append it to session history."""
        gemini_cfg = config.gemini
        user_turn = GeminiContent(role="user", parts=[GeminiPart(text=user_prompt_text)])
        trial_contents: list[GeminiContent] = [*self.gemini_contents, user_turn]

        llm_result = await call_gemini_text(
            contents=trial_contents,
            system_prompt=config.system_prompt,
            model=gemini_cfg.model,
            temperature=gemini_cfg.temperature,
            enable_google_search=gemini_cfg.enable_google_search,
        )

        audio_url: str | None = None
        if gemini_cfg.tts_enabled:
            audio_url = await call_gemini_tts(
                text=llm_result.text,
                tts_model=gemini_cfg.tts_model,
                voice_name=gemini_cfg.tts_voice,
                style_prompt=gemini_cfg.tts_style_prompt,
            )

        now = time.time()
        model_turn = GeminiContent(role="model", parts=[GeminiPart(text=llm_result.text)])
        self.gemini_contents = [*trial_contents, model_turn]
        self.last_trigger_time = now
        self.is_paused_audio = False
        self.paused_at = None
        self.paused_location = None
        if audio_url is None:
            self.last_audio_ended_at = now

        return llm_result, audio_url, now

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

        ref_time = max(
            (t for t in (self.last_audio_ended_at, self.last_trigger_time) if t is not None),
            default=None,
        )
        silence_elapsed_s = (now - ref_time) if ref_time is not None else None
        effective_cooldown_s = (
            min(min_cooldown_s, INITIAL_LOCATION_CHAT_GRACE_SECONDS)
            if self.last_trigger_location is None
            else min_cooldown_s
        )
        cooldown_remaining_s = 0.0
        if silence_elapsed_s is not None:
            cooldown_remaining_s = max(0.0, effective_cooldown_s - silence_elapsed_s)

        pause_hold_active = self.is_pause_hold_active(now)
        should_trigger = False
        trigger_reason: str | None = None

        if (
            auto_enabled
            and not self.is_generating
            and not self.is_playing_audio
            and not pause_hold_active
            and self.current_location
            and cooldown_remaining_s <= 0
        ):
            if self.last_trigger_location is None:
                should_trigger = True
                trigger_reason = "initial_location"
            elif dist_moved_m is not None and dist_moved_m >= min_dist_m:
                should_trigger = True
                trigger_reason = f"distance ({int(dist_moved_m)}m >= {int(min_dist_m)}m)"
            elif (
                silence_elapsed_s is not None
                and silence_elapsed_s >= max_silence_s
                and (dist_moved_m is None or dist_moved_m >= min_drift_m)
            ):
                should_trigger = True
                trigger_reason = f"max_silence ({int(silence_elapsed_s)}s >= {int(max_silence_s)}s)"

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
            is_paused_audio=pause_hold_active,
        )

    def should_prefetch_overpass(self, config: AppConfig, lat: float, lon: float) -> bool:
        """Return True if the user is approaching a narration trigger (>75% of min_distance)."""
        trig = config.triggers
        if (
            not trig.auto_trigger_enabled
            or self.is_generating
            or self.is_playing_audio
            or self.is_pause_hold_active(time.time())
        ):
            return False
        if self.last_trigger_location is None:
            return True
        dist_m = haversine_meters(
            lat,
            lon,
            self.last_trigger_location.lat,
            self.last_trigger_location.lon,
        )
        return dist_m >= 0.75 * trig.min_distance_meters

    @staticmethod
    def build_narration_prompt(
        lat: float,
        lon: float,
        heading: float | None,
        geo: GeoContext,
        lens: LensConfig,
        custom_focus: str | None = None,
    ) -> str:
        compass = compass_bearing_label(heading)
        nearby_str = (
            "\nNearby mapped features (within 150m): " + "; ".join(geo.nearby_features)
            if geo.nearby_features
            else ""
        )
        heading_str = f" | Heading: {compass}" if compass else ""
        focus_str = (
            f"\nAdditional user request for this stop: {custom_focus}" if custom_focus else ""
        )
        return (
            f"[LOCATION TRIGGER — {geo.short_label}]\n"
            f"Coordinates: {lat:.5f}, {lon:.5f}{heading_str}\n"
            f"Full address context: {geo.full_address}"
            f"{nearby_str}\n\n"
            f"Selected Lens ({lens.name}): {lens.prompt}"
            f"{focus_str}\n\n"
            "Narrate an insightful historical segment about this location "
            "through the selected lens."
        )

    async def execute_narration_turn(
        self,
        config: AppConfig,
        lat: float,
        lon: float,
        heading: float | None,
        geo: GeoContext,
        requested_lens_id: str | None = None,
        custom_focus: str | None = None,
        reason: str | None = "manual",
    ) -> AssistantMessage:
        lens = self.sample_lens(config, requested_lens_id=requested_lens_id)
        location_prompt = self.build_narration_prompt(
            lat=lat,
            lon=lon,
            heading=heading,
            geo=geo,
            lens=lens,
            custom_focus=custom_focus,
        )
        llm_result, audio_url, now = await self.run_gemini_turn(config, location_prompt)

        self.last_trigger_location = TriggerLocation(
            lat=lat,
            lon=lon,
            short_label=geo.short_label,
            timestamp=now,
        )

        msg_record = AssistantMessage(
            id=uuid.uuid4().hex[:10],
            type="narration",
            timestamp=now,
            location=MessageLocation(
                lat=lat,
                lon=lon,
                short_label=geo.short_label,
                nearby_features=geo.nearby_features,
            ),
            lens=LensSummary(id=lens.id, name=lens.name),
            reason=reason,
            model=config.gemini.model,
            text=llm_result.text,
            audio_url=audio_url,
            sources=llm_result.sources,
            search_queries=llm_result.search_queries,
        )
        self.ui_messages.append(msg_record)
        await asyncio.to_thread(self.persist_transcript)
        return msg_record

    async def execute_chat_turn(
        self,
        config: AppConfig,
        cleaned_message: str,
        lat: float | None,
        lon: float | None,
        short_label: str | None,
    ) -> tuple[UserChatMessage, AssistantMessage]:
        loc_prefix = (
            f"[User current location: {short_label} ({lat:.5f}, {lon:.5f})]\n"
            if short_label and lat is not None and lon is not None
            else ""
        )
        user_msg_ui = UserChatMessage(
            id=uuid.uuid4().hex[:10],
            timestamp=time.time(),
            location_label=short_label,
            text=cleaned_message,
        )
        user_prompt_text = f"{loc_prefix}User message: {cleaned_message}"
        llm_result, audio_url, now = await self.run_gemini_turn(config, user_prompt_text)

        assistant_msg_ui = AssistantMessage(
            id=uuid.uuid4().hex[:10],
            type="chat_reply",
            timestamp=now,
            location=(
                MessageLocation(lat=lat, lon=lon, short_label=short_label)
                if short_label and lat is not None and lon is not None
                else None
            ),
            model=config.gemini.model,
            text=llm_result.text,
            audio_url=audio_url,
            sources=llm_result.sources,
            search_queries=llm_result.search_queries,
        )
        self.ui_messages.append(user_msg_ui)
        self.ui_messages.append(assistant_msg_ui)
        await asyncio.to_thread(self.persist_transcript)
        return user_msg_ui, assistant_msg_ui

    def persist_transcript(self) -> None:
        out_file = SESSIONS_DIR / f"session_{self.session_id}.json"
        transcript = SessionTranscript(
            session_id=self.session_id,
            started_at=self.started_at,
            messages=self.ui_messages,
        )
        out_file.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")


class SessionManager:
    """Manages isolated WalkSession instances per client_id."""

    def __init__(self, max_sessions: int = MAX_ACTIVE_SESSIONS) -> None:
        self._sessions: dict[str, WalkSession] = {}
        self._max_sessions = max_sessions

    def get(self, raw_client_id: str | None) -> WalkSession:
        client_id = sanitize_client_id(raw_client_id)
        walk_session = self._sessions.get(client_id)
        if walk_session is None:
            if len(self._sessions) >= self._max_sessions:
                oldest_key = min(
                    self._sessions,
                    key=lambda k: self._sessions[k].last_active_at,
                )
                self._sessions.pop(oldest_key, None)
            walk_session = WalkSession(client_id=client_id)
            self._sessions[client_id] = walk_session
        walk_session.last_active_at = time.time()
        return walk_session


session_manager = SessionManager()
