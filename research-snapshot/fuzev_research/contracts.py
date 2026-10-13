import math
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Primitive(StrEnum):
    CHOICE = "choice"
    NOUL = "noul"
    ORDINAL = "ordinal"


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1)
    ordinal_value: float | None = None


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    primitive: Primitive
    question: str = Field(min_length=1)
    candidates: list[Candidate] = Field(min_length=2, max_length=16)
    prefix_id: str | None = None
    messages: list[dict] | None = None
    additional_context: str = ""
    head_id: str = "reference"

    @model_validator(mode="after")
    def validate_semantics(self):
        ids = [candidate.id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("Candidate identifiers must be unique")
        if self.primitive == Primitive.NOUL:
            if set(ids) != {"yes", "no"}:
                raise ValueError("Noul requires exactly the yes and no candidate identifiers")
            if any(candidate.text.casefold() != candidate.id for candidate in self.candidates):
                raise ValueError("Noul candidate text must match its Yes/No identifier")
        if self.primitive == Primitive.ORDINAL:
            values = [c.ordinal_value for c in self.candidates]
            if any(value is None or not math.isfinite(value) for value in values) or any(a >= b for a, b in zip(values, values[1:])):
                raise ValueError("Ordinal candidates require strictly increasing values")
        if (self.prefix_id is None) == (self.messages is None):
            raise ValueError("Provide either standalone messages or an existing prefix, not both")
        return self


class Score(BaseModel):
    id: str
    logit: float
    raw_probability: float
    calibrated_probability: float


class Execution(BaseModel):
    mode: Literal["fresh", "replay", "checkpoint", "reuse"]
    logical_prefix_tokens: int
    reused_tokens: int
    replayed_tokens: int
    forwarded_tokens: int
    consumed_tokens: int
    feature_tap: str = "final-mixer/post-collapse/before-lm-head"


class DecisionResponse(BaseModel):
    id: str
    model: str
    primitive: Primitive
    head_id: str
    calibration_id: str
    selected_id: str
    scores: list[Score]
    execution: Execution
