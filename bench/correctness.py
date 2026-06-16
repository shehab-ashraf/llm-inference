"""
correctness test: mutual top-k agreement.
both models generate independently (greedy), collecting top-k logprobs as
{token_id: logprob} dicts. on token mismatch: assert ref's sampled token is in
our top-k, and our sampled token is in ref's top-k.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# config
model_path = "./Qwen3-0.6B"
max_new_tokens = 64
top_k = 5
seed = 42

prompts = [
    "The capital of Egypt is",
    "What is the square root of 144?",
    "Translate to Spanish: Good morning.",
    "def binary_search(arr, target):",
    "What is the difference between a list and a tuple in Python?",
    "Explain gravity to a 10 year old.",
    "Write a C function that reverses a linked list in-place:",
    "A snail climbs 3 feet up a 10 foot wall each day but slides back 2 feet each night. How many days to reach the top? Think step by step.",
    "Below is buggy Python:\n```python\ndef flatten(lst):\n    result = []\n    for item in lst:\n        if type(item) == list:\n            result + flatten(item)\n        else:\n            result.append(item)\n    return result\n```\nWhat is the bug? Fix it.",
    "Explain the difference between a process and a thread, and when you would use each.",
]
# -----------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.qwen3 import Qwen3Model
from src.utils.load_utils import load_weights, apply_weights, load_config

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

# load
print(f"loading models from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=dtype, attn_implementation="sdpa"
).to(device).eval()

model_config = load_config(model_path)
model_config["dtype"] = dtype
our_model = Qwen3Model(model_config)
apply_weights(our_model, load_weights(model_path), model_config)
our_model = our_model.to(device).eval()

print(f"\ncorrectness — mutual top-{top_k} agreement\n")

# -----------------------------------------------------------------------------
# greedy generation

def greedy_generate(forward_fn, input_ids, max_tokens, top_k):
    tokens = []
    logprobs = []
    seq = input_ids
    for _ in range(max_tokens):
        logits = forward_fn(seq)
        last_logits = logits[0, -1, :]
        logp = F.log_softmax(last_logits.float(), dim=-1)
        topk_logp, topk_indices = torch.topk(logp, top_k)
        next_id = topk_indices[0].item()
        tokens.append(next_id)
        logprobs.append(dict(zip(topk_indices.tolist(), topk_logp.tolist())))
        seq = torch.cat([seq, torch.tensor([[next_id]], device=device)], dim=1)
    return tokens, logprobs

# -----------------------------------------------------------------------------

failed = False

with torch.inference_mode():
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        ref_tokens, ref_logprobs = greedy_generate(
            lambda x: ref_model(x).logits, input_ids, max_new_tokens, top_k
        )
        our_tokens, our_logprobs = greedy_generate(
            our_model, input_ids, max_new_tokens, top_k
        )

        ok = True
        n_steps = min(len(ref_tokens), len(our_tokens))

        for i in range(n_steps):
            if ref_tokens[i] == our_tokens[i]:
                continue

            if ref_tokens[i] not in our_logprobs[i] or our_tokens[i] not in ref_logprobs[i]:
                ok = False
            break

        label = "ok" if ok else "FAIL"
        print(f"{label:>4} | {prompt[:60]}...")
        if not ok:
            failed = True

if failed:
    sys.exit(1)
print("\nall ok.")
