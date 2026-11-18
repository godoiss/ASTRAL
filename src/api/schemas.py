"""Pydantic contracts for the versioned HTTP boundary."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Forbid undocumented request and response fields."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(StrictModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    stage: int
    local_only: bool
    models: dict[str, bool]
    artifacts: dict[str, bool]


class ApiInfoResponse(StrictModel):
    name: str
    version: str
    stage: int
    api_version: str
    documentation: dict[str, str]
    scientific_guardrails: dict[str, bool]


class Pagination(StrictModel):
    offset: int
    limit: int
    returned: int
    total: int


class PageResponse(StrictModel):
    items: list[dict[str, Any]]
    pagination: Pagination


class ScientificResponse(StrictModel):
    data: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class InferenceRequest(StrictModel):
    object_id: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9 ._-]+$")

    @field_validator("object_id")
    @classmethod
    def normalize_object_id(cls, value: str) -> str:
        return value.strip()


class InferenceResponse(StrictModel):
    model_id: Literal["ann-01", "ann-02"]
    model_experiment_id: str
    object_id: str
    kic: int
    original_disposition: str
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    predicted_class: str
    input_kind: str
    details: dict[str, Any]
    scientific_notice: str


class ErrorBody(StrictModel):
    code: str
    message: str
    details: list[dict[str, Any]] | None = None
    request_id: str | None = None


class ErrorResponse(StrictModel):
    error: ErrorBody
