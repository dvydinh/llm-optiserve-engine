"""LLM OptiServe Engine — Application entrypoint."""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI

from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="LLM OptiServe Engine",
    description=(
        "High-throughput LLM inference API built on vLLM (PagedAttention) "
        "with AWQ INT4 quantization for optimal GPU memory efficiency."
    ),
    version="1.0.0",
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
