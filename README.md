# LLM OptiServe Engine

> High-throughput LLM inference server leveraging **vLLM AsyncLLMEngine (PagedAttention)** and **AWQ INT4 quantization** for optimal GPU memory efficiency and low-latency serving.

---

## Architecture

![System Architecture](docs/architecture.png)

| Component | Role |
|---|---|
| **FastAPI Server** | Exposes `/generate` and `/health` endpoints with Pydantic request validation |
| **vLLM AsyncLLMEngine** | Serves quantized models using PagedAttention for iteration-level continuous batching and paged KV cache management |
| **AWQ Quantizer** | Compresses FP16 models to INT4 using activation-aware per-channel weight scaling |
| **Locust Load Tester** | Benchmarks throughput and latency under concurrent user load |

---

## Why AWQ over GPTQ?

AWQ and GPTQ both achieve INT4 weight quantization, but via fundamentally different mechanisms:

**GPTQ** uses Hessian-based **Optimal Brain Quantization (OBQ)** — for each layer, it computes second-order weight sensitivity (via the Hessian of the reconstruction loss), then iteratively quantizes columns while adjusting remaining weights to compensate. Complexity is O(d_row × d_col²) per layer, making it slow on large models.

**AWQ** takes a different approach based on a key insight from the paper: only ~1% of weight channels are "salient" (their quantization error dominates output loss). Instead of mixed-precision storage, AWQ applies a **per-channel scaling trick**:

1. For input channel `i` with activation magnitude `a_i`, compute scaling factor `s_i = a_i^α` (α ∈ [0,1], searched per-layer, typically ~0.5).
2. Scale the weight column: `w'_i = w_i × s_i` **before** quantization.
3. Compensate on activation: `x'_i = x_i / s_i`.
4. The product `W·x = (W·S)·(S⁻¹·x)` is mathematically identical, but salient weights are amplified into a range where INT4 rounding error is relatively smaller.

| Criteria | AWQ | GPTQ |
|---|---|---|
| Quantization speed | **Fast** — closed-form scaling, no iterative reconstruction | Slow — Hessian-based OBQ per layer |
| Quality at INT4 | Higher on ≥13B — preserves salient channels via activation scaling | Good, but can degrade on larger models |
| Calibration data | Minimal (~128 samples) | Requires larger calibration sets |
| Kernel support | Optimized GEMM/GEMV kernels | Requires Marlin or AutoGPTQ kernels |

## Why PagedAttention?

Traditional KV cache management pre-allocates a contiguous memory block per sequence for the maximum possible length. This causes:
- **Internal fragmentation**: reserved but unused memory within each sequence's allocation.
- **External fragmentation**: small gaps between allocations that are too small to reuse.

On real workloads, this wastes 60-80% of KV cache memory (measured in the original vLLM paper against HuggingFace Transformers' naive allocation on parallel sampling workloads).

**PagedAttention** (Kwon et al., 2023) fixes this by managing KV cache like OS virtual memory:
- KV data is split into fixed-size **blocks** (e.g., 16 tokens per block).
- A **block table** maps each sequence's logical blocks to non-contiguous physical GPU memory blocks.
- Blocks are allocated **on demand** and freed immediately when a sequence finishes.
- The block table enables **copy-on-write**: parallel sampling / beam search candidates can share KV cache blocks from the common prefix, duplicating only on divergence.
- **Prefix sharing**: multiple chat sessions with the same system prompt can share their prefix KV blocks in GPU memory.

---

## Expected Performance Characteristics

> **Important**: The numbers below are **theoretical estimates** based on published benchmarks from the vLLM and AWQ papers, not measurements from this specific deployment. Actual performance varies significantly with hardware, model, batch size, prompt length, and concurrent load. **Always benchmark on your own hardware.**

Estimated ranges for **Llama-2-7B-Chat** on a single GPU:

| Metric | HuggingFace FP16 (Baseline) | vLLM + AWQ INT4 | Notes |
|---|---|---|---|
| Model weights VRAM | ~13.4 GB | ~3.5 GB (weights only) | 6.7B params × 2B (FP16) vs × 0.5B (INT4) |
| Total VRAM (serving) | ~16-18 GB | ~5.5-7 GB | Includes KV cache, activations, CUDA context |
| Throughput (tokens/s) | ~200-400 | ~1,200-2,000 | Depends on batch size and sequence length |
| Latency (256 tok, single) | ~4-8s | ~2-4s | Single request, no batching |

**VRAM calculation breakdown** (INT4, Llama-2-7B):
- Weights: 6.7B × 4 bits = **3.35 GB**
- KV cache (4096 ctx, 1 seq): 2 × 32 layers × 32 heads × 128 dim × 4096 × FP16 ≈ **2 GB**
- CUDA context + activations + overhead: **~0.5-1 GB**
- Total: **~5.5-7 GB** depending on batch size and context usage

---

## Project Structure

```
llm-optiserve-engine/
├── api/
│   ├── __init__.py
│   ├── routes.py           # FastAPI endpoint definitions
│   └── schemas.py          # Pydantic request/response models
├── core/
│   ├── __init__.py
│   ├── engine.py           # vLLM AsyncLLMEngine wrapper
│   └── quantizer.py        # AWQ quantization script
├── load_tests/
│   └── locustfile.py       # Locust load testing scenarios
├── docs/
│   └── architecture.png    # System architecture diagram
├── main.py                 # Application entrypoint (lifespan manager)
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA 12.1+ and driver ≥ 530
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### 1. Build the Docker image

```bash
docker build -t llm-optiserve .
```

### 2. Run the server

```bash
docker run --gpus all \
  -p 8000:8000 \
  -v /path/to/models:/app/models \
  --env-file .env \
  llm-optiserve
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing in simple terms.", "max_tokens": 256}'
```

### 4. Load testing

```bash
pip install locust==2.24.1
locust -f load_tests/locustfile.py --host http://localhost:8000
```

---

## Local Development (without Docker)

```bash
conda create -n llm_serve python=3.10 -y
conda activate llm_serve
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your model path and GPU settings

python main.py
```

---

## AWQ Quantization

To quantize a model from FP16 to INT4:

```bash
# Configure source model and output path in .env, then:
python -m core.quantizer
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME_OR_PATH` | `TheBloke/Llama-2-7B-Chat-AWQ` | HF model ID or local path |
| `QUANTIZATION_METHOD` | `None` | Quantization format: `awq`, `gptq`, or unset for FP16 |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Fraction of VRAM allocated to the model |
| `MAX_MODEL_LEN` | `4096` | Maximum sequence length |
| `DTYPE` | `auto` | Model data type (`auto`, `float16`, `bfloat16`) |

See `.env.example` for the full list including quantizer settings.

---

## References

- Lin, J. et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. MLSys 2024.
- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023.

---

## Tech Stack

- **[vLLM](https://github.com/vllm-project/vllm)** — PagedAttention-based async inference engine
- **[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)** — Activation-aware weight quantization
- **[FastAPI](https://fastapi.tiangolo.com/)** — High-performance async API framework
- **[Locust](https://locust.io/)** — Distributed load testing

---

## License

MIT
