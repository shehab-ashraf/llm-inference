# llm-inference

Minimal LLM inference engine, built from scratch.  
Currently **Naive implementation** [no optimizations, pure PyTorch].

## What's here

Qwen3-0.6B inference implementation in ~500 lines:

```text
src/
├── models/qwen3.py       # model
├── utils/load_utils.py   # weight loading
├── config.py             # engine configuration
├── sampling_params.py    # sampling config
├── sampler.py            # Gumbel-max token sampling
└── llm.py                # engine
bench/
├── perf.py               # Performance benchmark
└── ppl.py                # perplexity vs HuggingFace
```

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Download model
hf download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B/
```

## Benchmarks

All benchmarks on **NVIDIA L4 (24GB)** via **Lightning AI**.

### Performance

```bash
python -m bench.perf
```

```
Qwen3-0.6B | 596M params | torch.bfloat16 | cuda
  28L / 16H / 128D | loaded in 10.3s | 1.13 GB VRAM

Warmup...

Benchmark
Prompt length: 128 | Max tokens: 256

Batch | Time (s) | TTFT (s) | TPOT (s) | VRAM (GB) |    TPS
------------------------------------------------------------
    1 |    9.209 |    0.039 |    0.036 |     1.355 |   27.8
    4 |   12.283 |    0.045 |    0.048 |     2.007 |   83.4
    8 |   24.653 |    0.045 |    0.097 |     2.876 |   83.1
   16 |   56.850 |    0.091 |    0.223 |     4.614 |   72.0
   32 |  141.093 |    0.227 |    0.552 |     8.090 |   58.1
   64 |  314.024 |    0.569 |    1.229 |    15.041 |   52.2
```

### Perplexity

```bash
python -m bench.ppl
```

```
>> hf ppl: 18.1799
>> model ppl: 18.1612
difference: 0.018715
relative: 0.1029%
```