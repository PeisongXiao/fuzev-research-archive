"""Historical pre-1.0 research service, superseded by fuzev-vllm-plugin and fuzev-exllamav3-plugin. It does not run against the published engine forks."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import DecisionRequest


class TrustedImage(BaseModel):
    """An acquired PNG, with ImageIndex's size-prefixed RGB pixel digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(min_length=1, max_length=4096)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ServiceDecisionRequest(DecisionRequest):
    # Standalone: canonical context image. Prefix: new additional evidence in C.
    image: TrustedImage | None = None


class TextMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str
    prompt: str | None = Field(default=None, min_length=1)
    messages: list[TextMessage] | None = Field(default=None, min_length=1)
    image: TrustedImage | None = None
    max_tokens: int = Field(default=256, ge=1, le=262144)
    temperature: float = Field(default=1., ge=0, le=2)
    top_p: float = Field(default=.95, gt=0, le=1)
    top_k: int = Field(default=20, ge=0)
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    stop: list[str] = Field(default_factory=list, max_length=4)
    stream: Literal[False] = False

    @model_validator(mode="after")
    def context_and_stops(self):
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("Provide exactly one of prompt or text messages")
        if any(not value or len(value) > 256 for value in self.stop):
            raise ValueError("Stops must contain 1 to 256 characters")
        return self


class ServiceConfig(BaseModel):
    """Operator-owned pins, never supplied by an inference request.

    The separate source pins explicitly acknowledge that serving code differs
    from extraction code. They do not assert numerical or quality qualification.
    Cross-bitrate use additionally requires an exact actual serving identity;
    feature_identity always identifies the unchanged frozen reference head.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    head_directory: str
    head_id: str = "reference"
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_identity: dict
    service_sources: dict[str, str | None]
    serving_identity: dict | None = Field(default=None, min_length=1)
    prefix_capacity: int = Field(default=8, ge=1, le=32)
    prefix_ttl_seconds: float = Field(default=900, ge=1, le=3600)
