"""AWQ quantization utility for compressing LLM weights.

Activation-aware Weight Quantization (AWQ) identifies salient weight channels
by observing activation distributions, then applies mixed-precision quantization
to preserve accuracy while reducing VRAM usage by ~3-4x compared to FP16.

Key advantage over GPTQ:
    AWQ does NOT require a costly round-to-nearest or iterative GPTQ-style
    reconstruction pass per layer. Instead, it scales weights based on
    activation awareness, yielding faster quantization and better quality
    retention at INT4, especially for larger models (13B+).

Usage:
    python -m core.quantizer
"""

from __future__ import annotations

import logging
import os

from awq import AutoAWQForCausalLM
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

logger = logging.getLogger(__name__)


def quantize_model(
    source_model_path: str | None = None,
    output_path: str | None = None,
    group_size: int | None = None,
    zero_point: bool | None = None,
    w_bit: int = 4,
    version: str | None = None,
    calib_size: int | None = None,
) -> str:
    """Run AWQ quantization on a source HuggingFace model.

    All parameters fall back to environment variables defined in ``.env``
    if not explicitly provided.

    Args:
        source_model_path: HuggingFace model ID or local path to the FP16 model.
        output_path: Directory to save the quantized model artifacts.
        group_size: Number of weight columns sharing a single scale factor.
        zero_point: Whether to use asymmetric quantization (zero-point).
        w_bit: Target bit width for weight quantization.
        version: AWQ kernel version (``gemm`` or ``gemv``).
        calib_size: Number of calibration samples from the dataset.

    Returns:
        The output path where quantized model was saved.
    """
    source_model_path = source_model_path or os.getenv(
        "SOURCE_MODEL_PATH",
        "meta-llama/Llama-2-7B-chat-hf",
    )
    output_path = output_path or os.getenv(
        "QUANTIZED_OUTPUT_PATH",
        "./models/llama-2-7b-chat-awq",
    )
    group_size = group_size if group_size is not None else int(
        os.getenv("AWQ_GROUP_SIZE", "128"),
    )
    zero_point = zero_point if zero_point is not None else (
        os.getenv("AWQ_ZERO_POINT", "true").lower() == "true"
    )
    version = version or os.getenv("AWQ_VERSION", "gemm")
    calib_size = calib_size if calib_size is not None else int(
        os.getenv("CALIBRATION_SIZE", "128"),
    )

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": w_bit,
        "version": version,
    }

    logger.info("Loading source model from '%s'...", source_model_path)
    model = AutoAWQForCausalLM.from_pretrained(source_model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        source_model_path,
        trust_remote_code=True,
    )

    logger.info(
        "Starting AWQ quantization (w_bit=%d, group_size=%d, calib_size=%d)...",
        w_bit,
        group_size,
        calib_size,
    )
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data="pileval",
        calib_data_config={"split": "train", "text_column": "text"},
        n_samples=calib_size,
    )

    logger.info("Saving quantized model to '%s'...", output_path)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info("Quantization complete.")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    quantize_model()
