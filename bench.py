"""Benchmark for LLM inference."""

import time
import torch
from random import randint, seed
from src.llm import LLM
from src.sampling_params import SamplingParams

# -----------------------------------------------------------------------------
# config
NUM_SEQS = 4           # run sequentially
MAX_INPUT_LEN = 64     # full recompute each step
MAX_OUTPUT_LEN = 64    # keep short sequences
SEED = 42              # random seed for reproducibility

# -----------------------------------------------------------------------------
# setup
seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

# LLM
llm = LLM(model_path="./Qwen3-0.6B", device=device, dtype=dtype)

# Random prompts as token IDs (converted to text via tokenizer)
prompt_lengths = [randint(8, MAX_INPUT_LEN) for _ in range(NUM_SEQS)]
prompts = [
    llm.tokenizer.decode([randint(0, 10000) for _ in range(length)])
    for length in prompt_lengths
]
output_lengths = [randint(16, MAX_OUTPUT_LEN) for _ in range(NUM_SEQS)]

# warmup
print("\nWarmup run...")
llm.generate("Hello", SamplingParams(max_tokens=5))

# -----------------------------------------------------------------------------
# benchmark
print(f"\nBenchmarking {NUM_SEQS} sequences...")
print(f"  Input:  8–{MAX_INPUT_LEN} tokens")
print(f"  Output: 16–{MAX_OUTPUT_LEN} tokens")
print()

total_generated = 0
total_time = 0.0

for i, (prompt, max_tokens) in enumerate(zip(prompts, output_lengths)):
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.6, top_k=50, top_p=0.95, min_p=0.05)
    input_len = len(llm.tokenizer.encode(prompt))

    t0 = time.perf_counter()
    output = llm.generate(prompt, sp)
    dt = time.perf_counter() - t0

    output_len = len(llm.tokenizer.encode(output[0]["text"]))
    tok_per_sec = output_len / dt if dt > 0 else 0
    total_generated += output_len
    total_time += dt

    print(f"  seq {i+1}/{NUM_SEQS}  "
          f"in={input_len:>4d} tok  out={output_len:>4d} tok  "
          f"{dt:>6.2f}s  {tok_per_sec:>6.1f} tok/s")

# -----------------------------------------------------------------------------
# summary
print()
avg_throughput = total_generated / total_time if total_time > 0 else 0
print(f"Total: {total_generated} tokens in {total_time:.2f}s")
print(f"Throughput: {avg_throughput:.2f} tok/s")
print(f"Device: {device}  Dtype: {dtype}")

