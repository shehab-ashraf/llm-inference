# llm-inference

Minimal LLM inference engine, built from scratch step-by-step.  
Currently **Naive implementation** [no optimizations, pure PyTorch].

## What's here

Qwen3-0.6B inference implementation in ~400 lines of Python:

```
src/
├── models/qwen3.py       # full transformer (RoPE, GQA, RMSNorm, SwiGLU)
├── utils/load_utils.py   # weight loading (safetensors / .bin)
├── sampling_params.py    # sampling config
└── llm.py                # LLM wrapper (tokenizer + model + generate)
example.py                # quick demo
bench.py                  # throughput benchmark
```

## Setup

```bash
pip install -r requirements.txt
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B/
```
## Run

```bash
python example.py
```

```
Qwen3-0.6B | 752M params | torch.bfloat16 | cuda
  28L / 16H / 128D | loaded in 12.0s

> Hello, 
 I'm trying to find the value of the integral from 0 to 1 of x^2 e^{-x} dx
  [25 tokens, 1.18s, 21.2 tok/s]
```

## Benchmark

```bash
python bench.py
```
```
Qwen3-0.6B | 752M params | torch.bfloat16 | cuda
  28L / 16H / 128D | loaded in 18.2s

Warmup...

Benchmark
Prompt length: 128 | Max tokens: 256
Device: cuda | Dtype: torch.bfloat16

   Batch |   Time (s) |   Tokens |   Throughput
--------------------------------------------------
       1 |     15.696 |      256 |         16.3
       4 |     16.701 |     1024 |         61.3
       8 |     25.722 |     2048 |         79.6
      16 |     56.960 |     4096 |         71.9
      32 |    141.323 |     8192 |         58.0
      64 |    313.775 |    16384 |         52.2
```

> This benchmark was run on an **NVIDIA L4 (24GB)** via **Lightning AI**.
