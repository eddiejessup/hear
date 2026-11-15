from pydantic import BaseModel, ConfigDict


class HearModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
        ser_json_inf_nan="constants",
    )
