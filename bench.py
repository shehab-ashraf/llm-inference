"""Inference benchmark."""

import time
import torch
from src.llm import LLM
from src.sampling_params import SamplingParams

# -----------------------------------------------------------------------------
# config

PROMPT_LEN = 128
MAX_TOKENS = 256
BATCH_SIZES = [1, 4, 8, 16, 32, 64]
SEED = 42

# -----------------------------------------------------------------------------
# setup

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

llm = LLM(model_path="./Qwen3-0.6B", device=device, dtype=dtype)

def create_prompts(batch_size, prompt_len):
    return torch.randint(1000, 10000, (batch_size, prompt_len))

# -----------------------------------------------------------------------------
# warmup

print("\nWarmup...")
warmup_tokens = create_prompts(1, 10).to(device)
llm.generate(warmup_tokens, SamplingParams(max_tokens=5, temperature=0.0))

# -----------------------------------------------------------------------------
# benchmark

print(f"\nBenchmark")
print(f"Prompt length: {PROMPT_LEN} | Max tokens: {MAX_TOKENS}\n")
print(f"{'Batch':>5} | {'Time (s)':>8} | {'TTFT (s)':>8} | {'TPOT (s)':>8} | {'VRAM (GB)':>9} | {'TPS':>6}")
print("-" * 59)

results = []
for batch_size in BATCH_SIZES:
    token_ids = create_prompts(batch_size, PROMPT_LEN)
    sp = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    outputs = llm.generate(token_ids, sp)
    dt = time.perf_counter() - t0

    ttft = outputs.ttft
    tpot = outputs.tpot
    vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    total_tokens = batch_size * MAX_TOKENS
    tps = total_tokens / dt

    results.append({
        "batch_size": batch_size,
        "time_s": dt,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "vram_gb": vram_gb,
        "tps": tps
    })

    print(f"{batch_size:>5} | {dt:>8.3f} | {ttft:>8.3f} | {tpot:>8.3f} | {vram_gb:>9.3f} | {tps:>6.1f}")
