import torch
from pathlib import Path
from safetensors import safe_open

def load_weights(model, checkpoint_dir, device="cpu"):
    path = Path(checkpoint_dir)
    state_dict = {}
    
    # 1. Load weights (safetensors preferred)
    files = sorted(path.glob("*.safetensors"))
    if not files:
        files = sorted(path.glob("pytorch_model*.bin"))
    
    if not files:
        raise FileNotFoundError(f"No weights found in {path}")
        
    for file in files:
        if file.suffix == ".safetensors":
            with safe_open(file, framework="pt", device=device) as f:
                for k in f.keys(): state_dict[k] = f.get_tensor(k)
        else:
            state_dict.update(torch.load(file, map_location=device, mmap=True))
            
    # 2. Map keys (HF -> Ours)
    # HF usually prefixes with "model.", we strip it.
    final_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."): k = k[6:]
        final_dict[k] = v
        
    # 3. Load into model
    model.load_state_dict(final_dict, strict=False)
    
    # 4. Tie weights (TinyStories etc)
    if not any(k.endswith("lm_head.weight") for k in final_dict):
        model.lm_head.weight = model.embed_tokens.weight
