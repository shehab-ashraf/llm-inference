# llm-inference

Minimal LLM inference engine, built from scratch.  
Currently **naive baseline** — no KV cache, no optimizations.

## Structure

```text
src/
├── models/qwen3.py       # transformer (RoPE, GQA, RMSNorm, SwiGLU)
├── utils/load_utils.py   # weight loading (safetensors)
├── sampling_params.py    # sampling config
├── sampler.py            # token sampling
└── llm.py                # engine (fast rust tokenizers + benchmark path)
bench.py                  # pure tensor throughput benchmark
```

## Setup

```bash
pip install -r requirements.txt
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B/
```

## Benchmark

```bash
python bench.py
```

```
Qwen3-0.6B | 752M params | torch.bfloat16 | cuda
  28L / 16H / 128D | loaded in 11.4s | 1.42 GB VRAM

Warmup...

Benchmark
Prompt length: 128 | Max tokens: 256

Batch | Time (s) | TTFT (s) | TPOT (s) | VRAM (GB) |    TPS
-----------------------------------------------------------
    1 |    9.381 |    0.065 |    0.037 |     1.647 |   27.3
    4 |   12.019 |    0.045 |    0.047 |     2.299 |   85.2
    8 |   23.778 |    0.043 |    0.093 |     3.168 |   86.1
   16 |   56.430 |    0.092 |    0.221 |     4.906 |   72.6
   32 |  141.460 |    0.237 |    0.554 |     8.381 |   57.9
   64 |  316.827 |    0.569 |    1.240 |    15.333 |   51.7
```

> Benchmarked on **NVIDIA L4 (24GB)** via **Lightning AI**.
