"""Core inference engine wrapping vLLM AsyncLLMEngine for online serving.

Uses AsyncLLMEngine (not the offline LLM class) to avoid blocking the
FastAPI event loop. This is critical: vllm.LLM.generate() is synchronous
and would serialize all requests to throughput=1.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput

logger = logging.getLogger(__name__)


class InferenceEngine:
    """High-throughput async LLM inference engine backed by vLLM.

    Leverages PagedAttention for efficient KV cache memory management:
    the KV cache is split into fixed-size blocks tracked by a block table
    (logical block → physical GPU memory block mapping). This eliminates
    both internal fragmentation (pre-allocated but unused memory within a
    sequence's reservation) and external fragmentation (small unusable gaps
    between allocations). The block table also enables copy-on-write
    semantics for parallel sampling and beam search, and prefix sharing
    for system prompts across chat sessions.

    Attributes:
        model_name: HuggingFace model identifier or local path.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory reserved for the model.
        max_model_len: Maximum sequence length (prompt + generated tokens).
        quantization: Quantization method (e.g. 'awq', 'gptq', or None).
    """

    def __init__(
        self,
        model_name: str | None = None,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_model_len: int | None = None,
        dtype: str | None = None,
        quantization: str | None = None,
    ) -> None:
        self.model_name: str = model_name or os.getenv(
            "MODEL_NAME_OR_PATH",
            "TheBloke/Llama-2-7B-Chat-AWQ",
        )
        self.tensor_parallel_size: int = tensor_parallel_size or int(
            os.getenv("TENSOR_PARALLEL_SIZE", "1"),
        )
        self.gpu_memory_utilization: float = gpu_memory_utilization or float(
            os.getenv("GPU_MEMORY_UTILIZATION", "0.85"),
        )
        self.max_model_len: int = max_model_len or int(
            os.getenv("MAX_MODEL_LEN", "4096"),
        )
        self.dtype: str = dtype or os.getenv("DTYPE", "auto")
        self.quantization: str | None = quantization or os.getenv(
            "QUANTIZATION_METHOD",
            None,
        )

        self._engine: AsyncLLMEngine | None = None

    async def load_model(self) -> None:
        """Initialize the vLLM async engine with the configured model.

        The async engine spawns a background loop that continuously
        processes batched requests using iteration-level scheduling,
        enabling true concurrent throughput (unlike the offline LLM class).
        """
        logger.info(
            "Loading model '%s' (tp=%d, gpu_mem=%.2f, quant=%s)",
            self.model_name,
            self.tensor_parallel_size,
            self.gpu_memory_utilization,
            self.quantization or "none",
        )

        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            quantization=self.quantization,
            trust_remote_code=True,
        )

        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("Async engine initialized successfully.")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> RequestOutput:
        """Run async inference on a single prompt.

        The request is submitted to the engine's internal scheduler,
        which batches it with other concurrent requests at the
        iteration level — so multiple requests run truly in parallel
        on the GPU, not sequentially.

        Args:
            prompt: Input text to generate from.
            max_tokens: Maximum number of tokens to produce.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling cumulative probability cutoff.
            top_k: Top-k filtering parameter (-1 to disable).
            stop: Optional list of stop strings.

        Returns:
            A vLLM ``RequestOutput`` containing generated text and metadata.

        Raises:
            RuntimeError: If the engine has not been initialized.
        """
        if self._engine is None:
            raise RuntimeError(
                "Engine not initialized. Call load_model() before generate()."
            )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )

        request_id = str(uuid.uuid4())
        final_output: RequestOutput | None = None

        async for output in self._engine.generate(
            prompt, sampling_params, request_id
        ):
            final_output = output

        if final_output is None:
            raise RuntimeError("Engine returned no output for the request.")

        return final_output

    async def shutdown(self) -> None:
        """Abort all pending requests and release engine resources."""
        if self._engine is not None:
            self._engine.abort(request_id=None)
            logger.info("Engine shut down.")

    @property
    def is_ready(self) -> bool:
        """Check whether the engine has been initialized."""
        return self._engine is not None
