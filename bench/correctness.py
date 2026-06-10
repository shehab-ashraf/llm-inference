"""
correctness test
compares our custom model against huggingface using "mutual top-k agreement".
if top-1 tokens differ, we ensure hf's top-1 is in our top-k, and our top-1 is in hf's top-k.
"""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# configuration
model_path = "./Qwen3-0.6B"
max_tokens = 64
top_k = 3
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

# set up
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"loading hf model from {model_path}...")
tok = AutoTokenizer.from_pretrained(model_path)
hf = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
).to(device).eval()

print("loading custom model...")
model_config = load_config(model_path)
model_config["dtype"] = torch.bfloat16
model = Qwen3Model(model_config)
state_dict = load_weights(model_path)
apply_weights(model, state_dict, model_config)
model = model.to(device).eval()

print(f"\nrunning correctness checks (top-{top_k} agreement)...\n")

all_ok = True
with torch.inference_mode():
    for prompt in prompts:
        
        # generate reference sequence using hf
        inputs = tok(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        prompt_len = input_ids.shape[1]
        
        output = hf.generate(
            input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            do_sample=False, # greedy decode
            pad_token_id=tok.eos_token_id,
        )
        hf_seq = output[0] # full sequence [prompt + generated]

        # forward pass sequence to get logits from both models
        model_input = hf_seq[:-1].unsqueeze(0).to(dtype=torch.long)
        logits = model(model_input) # (1, seq_len, vocab_size)
        hf_logits = hf(model_input).logits

        # check mutual top-k agreement
        ok = True
        for i in range(prompt_len - 1, len(hf_seq) - 1):
            step_logits = logits[0, i]
            step_hf_logits = hf_logits[0, i]
            
            our_top_1 = step_logits.argmax().item()
            hf_top_1 = step_hf_logits.argmax().item()

            if our_top_1 != hf_top_1:
                our_top_k = step_logits.topk(top_k).indices.tolist()
                hf_top_k = step_hf_logits.topk(top_k).indices.tolist()
                
                if hf_top_1 not in our_top_k or our_top_1 not in hf_top_k:
                    ok = False
                    break

        # print result
        print(f"{'ok' if ok else 'fail'} | {repr(prompt[:60])}...")
        if not ok:
            print(f"  -> diff at step {i - prompt_len + 1}: hf={hf_top_1}, ours={our_top_1}")
            all_ok = False

if not all_ok:
    sys.exit(1)

print("\nall ok.")
