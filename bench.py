"""inference benchmark for LLM."""

import time
import torch
from src.llm import LLM
from src.sampling_params import SamplingParams

# config
PROMPT_LEN = 128
MAX_TOKENS = 256
BATCH_SIZES = [1, 4, 8, 16, 32, 64]
SEED = 42

# setup
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

# LLM
llm = LLM(model_path="./Qwen3-0.6B", device=device, dtype=dtype)

# Create fixed-length prompts
def create_prompts(batch_size, prompt_len):
    """Generate batch_size prompts of exactly prompt_len tokens."""
    token_ids = torch.randint(1000, 10000, (batch_size, prompt_len))
    return token_ids

# Warmup
print("\nWarmup...")
llm.generate("Hello world", SamplingParams(max_tokens=5, temperature=0.0))

# Benchmark
print(f"\nBenchmark")
print(f"Prompt length: {PROMPT_LEN} | Max tokens: {MAX_TOKENS}")
print(f"Device: {device} | Dtype: {dtype}\n")
print(f"{'Batch':>8} | {'Time (s)':>10} | {'Tokens':>8} | {'Throughput':>12}")
print("-" * 50)

for batch_size in BATCH_SIZES:
    token_ids = create_prompts(batch_size, PROMPT_LEN)
    sp = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_k=50,
        top_p=0.95,
        min_p=0.05
    )

    # Benchmark generation
    t0 = time.perf_counter()
    outputs = llm.generate(token_ids, sp)
    dt = time.perf_counter() - t0

    # Total tokens generated: batch_size * max_tokens
    total_tokens = batch_size * MAX_TOKENS

    throughput = total_tokens / dt if dt > 0 else 0
    print(f"{batch_size:>8} | {dt:>10.3f} | {total_tokens:>8} | {throughput:>12.1f}")

print()