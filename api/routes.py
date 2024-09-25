"""FastAPI route definitions for the LLM inference service."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException

from api.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from core.engine import InferenceEngine

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton engine instance shared across requests
engine = InferenceEngine()


@router.on_event("startup")
async def startup_load_model() -> None:
    """Load the model into GPU memory when the server starts."""
    logger.info("Server startup — initializing inference engine...")
    engine.load_model()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health_check() -> HealthResponse:
    """Return current service status and loaded model metadata."""
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return HealthResponse(
        status="healthy",
        model_name=engine.model_name,
        gpu_memory_utilization=engine.gpu_memory_utilization,
    )


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Generate text from a prompt",
)
async def generate_text(request: GenerateRequest) -> GenerateResponse:
    """Run inference on the provided prompt and return generated text.

    The underlying vLLM engine uses PagedAttention to manage the KV cache
    in paged, non-contiguous GPU memory blocks — preventing fragmentation
    and enabling higher concurrent throughput.
    """
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        output = engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
        )
    except Exception as exc:
        logger.exception("Inference failed for prompt: %.80s...", request.prompt)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    completion = output.outputs[0]

    return GenerateResponse(
        request_id=str(uuid.uuid4()),
        prompt=request.prompt,
        generated_text=completion.text,
        token_count=len(completion.token_ids),
        finish_reason=completion.finish_reason or "unknown",
    )
