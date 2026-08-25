# llm-inference


Minimal, high-performance LLM inference engine built from scratch in pure PyTorch.
## Architecture

<img src="image/engine.png" alt="Architecture" width="70%">



## Quick Start

### 1. Installation

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync --extra bench
source .venv/bin/activate

# Download model weights (Qwen3-0.6B)
hf download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B/
```

### 2. Basic Usage

```python
import torch
from src.llm import LLM, SamplingParams

# Initialize engine
llm = LLM(model_path="./Qwen3-0.6B", device="cuda", dtype=torch.bfloat16)

# Tokenized prompt IDs (B, S)
prompt_ids = torch.tensor([[151644, 872, 198, 2610, 525, 264, 10950, 151645]], device="cuda")

# Generate
sampling_params = SamplingParams(max_tokens=64, temperature=0.7)
result = llm.generate(prompt_ids, sampling_params)

print(f"Generated tokens shape: {result.output.shape}")
print(f"TTFT: {result.ttft * 1000:.2f} ms | TPOT: {result.tpot * 1000:.2f} ms")
```



## Benchmarks

Benchmarked on **NVIDIA L4 (24 GB)** (Prompt length 128, max tokens 256, greedy decoding).

### Performance

```bash
python -m bench.perf
```

```text
Qwen3-0.6B | 596M params | torch.bfloat16 | cuda
  28L / 16H / 128D | loaded in 7.6s | 1.13 GB VRAM

Batch | Time (s) | TTFT (s) | TPOT (s) | VRAM (GB) |    TPS
------------------------------------------------------------
    1 |    7.031 |    0.032 |    0.027 |     1.153 |   36.4
    4 |   10.123 |    0.033 |    0.040 |     1.199 |  101.2
    8 |   21.092 |    0.041 |    0.083 |     1.259 |   97.1
   16 |   50.554 |    0.079 |    0.198 |     1.379 |   81.0
   32 |  124.495 |    0.195 |    0.487 |     1.621 |   65.8
   64 |  275.020 |    0.497 |    1.077 |     2.104 |   59.6
```

### Correctness

Sliding-window perplexity on WikiText-2 test set vs. Hugging Face reference:

```bash
python -m bench.ppl
```

```text
>> hf ppl: 18.1799
>> model ppl: 18.1612
difference: 0.018715
relative: 0.1029%
```
