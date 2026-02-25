"""Simple example: LLM inference with Qwen3-0.6B."""

import time
import torch
from src.llm import LLM
from src.sampling_params import SamplingParams

# -----------------------------------------------------------------------------
# setup
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

llm = LLM(
    model_path="./Qwen3-0.6B",
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

# -----------------------------------------------------------------------------
# generate
prompts = ["Hello, "]
sampling_params = SamplingParams(max_tokens=25, temperature=0.6, top_k=50, top_p=0.95, min_p=0.05)

for prompt in prompts:
    t0 = time.perf_counter()
    output = llm.generate(prompt, sampling_params)
    dt = time.perf_counter() - t0
    text = output[0]["text"]
    n_tokens = len(llm.tokenizer.encode(text))
    print(f"\n> {prompt}")
    print(f"{text}")
    print(f"  [{n_tokens} tokens, {dt:.2f}s, {n_tokens/dt:.1f} tok/s]")