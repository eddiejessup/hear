from pydantic import BaseModel, ConfigDict


class HearModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
        allow_inf_nan=False,
    )
