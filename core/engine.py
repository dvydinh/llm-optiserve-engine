"""Core inference engine wrapping vLLM with PagedAttention for optimized serving."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from vllm import LLM, SamplingParams

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput

load_dotenv()

logger = logging.getLogger(__name__)


class InferenceEngine:
    """High-throughput LLM inference engine backed by vLLM.

    Leverages PagedAttention for efficient KV cache memory management,
    enabling higher batch concurrency without OOM errors.

    Attributes:
        model_name: HuggingFace model identifier or local path.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory reserved for the model.
        max_model_len: Maximum sequence length (prompt + generated tokens).
    """

    def __init__(self) -> None:
        self.model_name: str = os.getenv(
            "MODEL_NAME_OR_PATH",
            "TheBloke/Llama-2-7B-Chat-AWQ",
        )
        self.tensor_parallel_size: int = int(
            os.getenv("TENSOR_PARALLEL_SIZE", "1"),
        )
        self.gpu_memory_utilization: float = float(
            os.getenv("GPU_MEMORY_UTILIZATION", "0.85"),
        )
        self.max_model_len: int = int(
            os.getenv("MAX_MODEL_LEN", "4096"),
        )
        self.dtype: str = os.getenv("DTYPE", "float16")

        self._engine: LLM | None = None

    def load_model(self) -> None:
        """Initialize the vLLM engine with the configured model.

        The engine uses PagedAttention under the hood, which manages KV cache
        in non-contiguous memory pages — similar to OS virtual memory — to
        eliminate fragmentation and allow near-optimal GPU memory usage.
        """
        logger.info(
            "Loading model '%s' with tensor_parallel=%d, gpu_mem=%.2f",
            self.model_name,
            self.tensor_parallel_size,
            self.gpu_memory_utilization,
        )
        self._engine = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            quantization="awq",
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> RequestOutput:
        """Run inference on a single prompt.

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
            RuntimeError: If the engine has not been initialized via ``load_model``.
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

        outputs: list[RequestOutput] = self._engine.generate(
            [prompt],
            sampling_params,
        )
        return outputs[0]

    @property
    def is_ready(self) -> bool:
        """Check whether the engine has been initialized."""
        return self._engine is not None
