"""Perplexity benchmark."""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.qwen3 import Qwen3Model
from src.utils.load_utils import load_weights, apply_weights, load_config

# -----------------------------------------------------------------------------
# config

model_path = "./Qwen3-0.6B"
dataset_name = "wikitext"
dataset_config = "wikitext-2-raw-v1"
split = "test"
max_length = 2048
stride = 512
seed = 42

# -----------------------------------------------------------------------------
# setup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# sliding window perplexity

@torch.inference_mode()
def compute_ppl(get_logits, input_ids):
    seq_len = input_ids.size(1)
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        ids = input_ids[:, begin_loc:end_loc].to(device)

        target_ids = ids.clone()
        target_ids[:, :-trg_len] = -100  # mask overlap: only score new tokens

        logits = get_logits(ids)

        # next-token prediction: shift by 1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )

        valid = shift_labels.view(-1) != -100
        nll_sum += loss[valid].sum().item()
        n_tokens += valid.sum().item()

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.exp(torch.tensor(nll_sum / n_tokens)).item()

# -----------------------------------------------------------------------------
# load data

print(f"loading tokenizer and dataset '{dataset_name}'...")
tok = AutoTokenizer.from_pretrained(model_path)
test_data = load_dataset(dataset_name, dataset_config, split=split)
encodings = tok("\n\n".join(test_data["text"]), return_tensors="pt")
print(f"dataset loaded. total tokens: {encodings.input_ids.size(1)}")

# -----------------------------------------------------------------------------
# reference: huggingface model

print(f"\nloading hf model from {model_path}...")
hf = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
).to(device).eval()

print("computing hf perplexity...")
hf_ppl = compute_ppl(lambda ids: hf(ids).logits, encodings.input_ids)
print(f">> hf ppl: {hf_ppl:.4f}")

del hf
torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# custom model

print("\nloading custom model...")
model_config = load_config(model_path)
model_config["dtype"] = torch.bfloat16
model = Qwen3Model(model_config)
state_dict = load_weights(model_path)
apply_weights(model, state_dict, model_config)
model = model.to(device).eval()

print("computing model perplexity...")
engine_ppl = compute_ppl(lambda ids: model(ids), encodings.input_ids)
print(f">> model ppl: {engine_ppl:.4f}")

# -----------------------------------------------------------------------------
# compare

diff = abs(hf_ppl - engine_ppl)
print(f"\ndifference: {diff:.6f}")
print(f"relative: {100 * diff / hf_ppl:.4f}%")
