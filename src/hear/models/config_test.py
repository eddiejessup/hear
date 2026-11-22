from __future__ import annotations

import pytest
from pydantic import ValidationError

from hear.models.config import GeminiConfig


def test_gemini_config_model_id_validation() -> None:
    cfg = GeminiConfig()
    assert cfg.model == "gemini-4-hydrogen"
    assert cfg.thinking_level == "medium"

    with pytest.raises(ValidationError):
        GeminiConfig(model="../bad/model")
    with pytest.raises(ValidationError):
        GeminiConfig.model_validate({"thinking_level": "minimal"})
