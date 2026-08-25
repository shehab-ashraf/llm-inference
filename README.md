# llm-inference

Minimal LLM inference engine, built from scratch.  
Currently **Naive implementation** [no optimizations, pure PyTorch].

## What's here
Qwen3-0.6B inference implementation in ~500 lines:

```text
src/
├── models/qwen3.py       # transformer (RoPE, GQA, RMSNorm, SwiGLU)
├── utils/load_utils.py   # weight loading (safetensors / .bin, shape validation)
├── config.py             # engine configuration (dataclass)
├── sampling_params.py    # sampling config (dataclass + validation)
├── sampler.py            # FlashInfer-based token sampling
└── llm.py                # engine (tokenizer + model + generation)
bench/
├── perf.py               # throughput benchmark
├── ppl.py                # perplexity vs HuggingFace (Wikitext-2)
└── correctness.py        # mutual top-k agreement test vs HuggingFace
```


## Setup

```bash
# Install uv 
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download model
hf download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B/
```

## Benchmark

```bash
python -m bench.perf
```

```
Qwen3-0.6B | 752M params | torch.bfloat16 | cuda
  28L / 16H / 128D | loaded in 11.2s | 1.42 GB VRAM

Warmup...

Benchmark
Prompt length: 128 | Max tokens: 256

Batch | Time (s) | TTFT (s) | TPOT (s) | VRAM (GB) |    TPS
-----------------------------------------------------------
    1 |    9.017 |    0.042 |    0.035 |     1.645 |   28.4
    4 |   12.058 |    0.044 |    0.047 |     2.297 |   84.9
    8 |   23.866 |    0.043 |    0.093 |     3.166 |   85.8
   16 |   57.042 |    0.090 |    0.223 |     4.904 |   71.8
   32 |  142.074 |    0.237 |    0.556 |     8.379 |   57.7
   64 |  316.794 |    0.569 |    1.240 |    15.331 |   51.7
```

> Benchmarked on **NVIDIA L4 (24GB)** via **Lightning AI**.
