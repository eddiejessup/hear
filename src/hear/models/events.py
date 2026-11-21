from enum import StrEnum


class EventSource(StrEnum):
    BACKEND = "backend"
    FRONTEND = "frontend"


class EventLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class BackendEventType(StrEnum):
    SERVER_STARTED = "server_started"
    HTTP_REQUEST = "http_request"
    HTTP_ERROR = "http_error"
    HTTP_UNHANDLED_ERROR = "http_unhandled_error"
    AUDIO_ENDED_PING = "audio_ended_ping"
    AUTO_TRIGGER_READY = "auto_trigger_ready"
    CONFIG_UPDATED = "config_updated"
    SESSION_RESET = "session_reset"
    NARRATION_COMPLETED = "narration_completed"
    CHAT_COMPLETED = "chat_completed"
    GEO_NOMINATIM_OK = "geo_nominatim_ok"
    GEO_NOMINATIM_ERROR = "geo_nominatim_error"
    GEO_OVERPASS_OK = "geo_overpass_ok"
    GEO_OVERPASS_ERROR = "geo_overpass_error"
    GEMINI_TEXT_ERROR = "gemini_text_error"
    GEMINI_TTS_ERROR = "gemini_tts_error"


__all__ = ["BackendEventType", "EventLevel", "EventSource"]
