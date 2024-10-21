"""AWQ quantization utility for compressing LLM weights.

Activation-aware Weight Quantization (AWQ) preserves model quality at INT4 by
identifying salient weight channels — those whose quantization error has the
largest impact on output loss, as measured by activation magnitude.

Rather than keeping salient channels at higher precision (which would require
mixed-precision storage), AWQ applies a per-channel scaling trick:

    Given input channel i with activation magnitude a_i:
    1. Compute scaling factor s_i = (a_i)^alpha, where alpha ∈ [0,1] is
       searched to minimize quantization error (typically alpha ~ 0.5).
    2. Multiply weight column w_i by s_i BEFORE quantization.
    3. Divide the corresponding activation x_i by s_i to compensate.

    This is mathematically equivalent to W·x = (W·S) · (S^{-1}·x), so the
    output is unchanged, but the salient weights are amplified into a range
    where INT4 rounding error is relatively smaller.

Key advantage over GPTQ:
    GPTQ uses Hessian-based Optimal Brain Quantization (OBQ) to reconstruct
    each layer iteratively, minimizing squared error via second-order info.
    This is accurate but slow (O(d_row * d_col^2) per layer). AWQ's scaling
    approach is a closed-form operation — no iterative reconstruction needed,
    yielding 3-5× faster quantization with comparable or better perplexity
    on models ≥13B parameters.

Usage:
    python -m core.quantizer
"""

from __future__ import annotations

import logging
import os

from awq import AutoAWQForCausalLM
from dotenv import load_dotenv
from transformers import AutoTokenizer

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
    load_dotenv()

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
