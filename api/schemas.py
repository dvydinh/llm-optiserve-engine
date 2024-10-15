"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Schema for text generation request payload."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The input prompt for text generation.",
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher values produce more random output.",
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold.",
    )
    top_k: int = Field(
        default=50,
        ge=-1,
        le=1000,
        description="Top-k sampling parameter. Set to -1 to disable.",
    )
    stop: list[str] | None = Field(
        default=None,
        description="List of stop sequences to halt generation.",
    )


class GenerateResponse(BaseModel):
    """Schema for text generation response payload."""

    request_id: str = Field(
        ...,
        description="Unique identifier for this generation request.",
    )
    prompt: str = Field(
        ...,
        description="The original input prompt.",
    )
    generated_text: str = Field(
        ...,
        description="The generated output text.",
    )
    token_count: int = Field(
        ...,
        ge=0,
        description="Number of tokens generated.",
    )
    finish_reason: str = Field(
        ...,
        description="Reason for generation completion (e.g., 'stop', 'length').",
    )


class HealthResponse(BaseModel):
    """Schema for health check endpoint response."""

    status: str = Field(
        ...,
        description="Current service status.",
    )
    model_name: str = Field(
        ...,
        description="Name or path of the loaded model.",
    )
    gpu_memory_utilization: float = Field(
        ...,
        description="Configured GPU memory utilization fraction.",
    )


class ErrorResponse(BaseModel):
    """Schema for error response payload."""

    detail: str = Field(
        ...,
        description="Detailed error message.",
    )
