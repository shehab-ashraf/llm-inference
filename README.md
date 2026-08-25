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
  28L / 16H / 128D | loaded in 10.2s

> Hello,
  and thank you for your time.  I am a student who is interested in...
```

## Benchmark

```bash
python bench.py
```
```
Qwen3-0.6B | 752M params | torch.bfloat16 | cuda
  28L / 16H / 128D | loaded in 12.2s

Warmup run...

Benchmarking 4 sequences...
  Input:  8–64 tokens
  Output: 16–64 tokens

  seq 1/4  in=  50 tok  out=  59 tok    2.13s    27.6 tok/s
  seq 2/4  in=  15 tok  out=  50 tok    1.85s    27.1 tok/s
  seq 3/4  in=   9 tok  out=  64 tok    2.34s    27.3 tok/s
  seq 4/4  in=  55 tok  out=  33 tok    1.18s    27.9 tok/s

Total: 206 tokens in 7.50s
Throughput: 27.45 tok/s
Device: cuda  Dtype: torch.bfloat16
```

> This benchmark was run on an **NVIDIA L4 (24GB)** via **Lightning AI**.




> This is intentionally slow — full sequence recompute every token, no KV cache, no batching. That's the baseline we'll optimize from.
