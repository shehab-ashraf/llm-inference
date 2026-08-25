"""Simple loader for HuggingFace-style model checkpoints."""

import json
from pathlib import Path
import torch
from safetensors.torch import load_file


# -----------------------------------------------------------------------------
# Config

def load_config(model_path):
    """Read config.json from model dir; return as dict."""
    config_path = Path(model_path) / "config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Weights

def load_weights(model_path):
    """Load .safetensors or .bin; merge shards into one dict."""
    weight_files = list(Path(model_path).glob("*.safetensors"))
    if not weight_files:
        weight_files = list(Path(model_path).glob("*.bin"))
    if not weight_files:
        raise ValueError(f"No weight files found in {model_path} (*.safetensors or *.bin)")

    state_dict = {}
    for wf in weight_files:
        if wf.suffix == ".safetensors":
            state_dict.update(load_file(str(wf)))
        else:
            state_dict.update(torch.load(str(wf), map_location="cpu"))

    return state_dict


def apply_weights(model, state_dict, config):
    """Load checkpoint weights into model with shape validation."""
    def assign(param, tensor, name=""):
        if param.shape != tensor.shape:
            raise ValueError(f"Shape mismatch for {name}: model={param.shape}, ckpt={tensor.shape}")
        with torch.no_grad():
            param.copy_(tensor.to(param.dtype))

    # embedding
    assign(model.tok_emb.weight, state_dict["model.embed_tokens.weight"], "embed_tokens")

    # transformer layers
    for i in range(config["num_hidden_layers"]):
        block = model.blocks[i]
        attn = block.attn
        prefix = f"model.layers.{i}"

        # norms
        assign(block.norm1.scale, state_dict[f"{prefix}.input_layernorm.weight"], f"L{i}.input_norm")
        assign(block.norm2.scale, state_dict[f"{prefix}.post_attention_layernorm.weight"], f"L{i}.post_attn_norm")

        # attention projections
        for proj_name, param in [
            ("q_proj", attn.W_query), ("k_proj", attn.W_key),
            ("v_proj", attn.W_value), ("o_proj", attn.out_proj),
        ]:
            assign(param.weight, state_dict[f"{prefix}.self_attn.{proj_name}.weight"], f"L{i}.{proj_name}")

        # QK norms
        if config.get("qk_norm", True):
            q_key = f"{prefix}.self_attn.q_norm.weight"
            k_key = f"{prefix}.self_attn.k_norm.weight"
            if q_key in state_dict:
                assign(attn.q_norm.scale, state_dict[q_key], f"L{i}.q_norm")
            if k_key in state_dict:
                assign(attn.k_norm.scale, state_dict[k_key], f"L{i}.k_norm")

        # FFN
        for proj_name, param in [
            ("gate_proj", block.ff.gate_proj), ("up_proj", block.ff.up_proj),
            ("down_proj", block.ff.down_proj),
        ]:
            assign(param.weight, state_dict[f"{prefix}.mlp.{proj_name}.weight"], f"L{i}.{proj_name}")

    # final norm
    assign(model.final_norm.scale, state_dict["model.norm.weight"], "final_norm")

    # lm head (or tied embeddings)
    if "lm_head.weight" in state_dict:
        assign(model.lm_head.weight, state_dict["lm_head.weight"], "lm_head")
    else:
        model.lm_head.weight = model.tok_emb.weight
        print("Using tied embeddings (lm_head = tok_emb)")