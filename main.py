"""LLM OptiServe Engine — Application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from api.routes import router, engine

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle: load model on startup, cleanup on shutdown.

    This replaces the deprecated @app.on_event("startup") pattern and
    ensures the model is fully loaded before any request is served.
    """
    await engine.load_model()
    yield
    await engine.shutdown()


app = FastAPI(
    title="LLM OptiServe Engine",
    description=(
        "High-throughput LLM inference API built on vLLM (PagedAttention) "
        "with AWQ INT4 quantization for optimal GPU memory efficiency."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
