import time
import argparse
import torch
from pathlib import Path

from config.llama import get_config
from model.transformer import Transformer
from utils.hf_tokenizer import Tokenizer
from utils.weights import load_weights

def sample(logits, temperature=0.7, top_k=50):
    # Get last token logits: [batch, vocab]
    logits = logits[:, -1, :]
    
    # Apply temperature
    if temperature > 0:
        logits /= temperature
        
    # Apply top-k
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to model weights")
    parser.add_argument("--config", required=True, help="Model config name")
    parser.add_argument("--prompt", default="Once upon a time", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)

    args = parser.parse_args()

    # 1. Setup
    if torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    path = Path(args.model_path)
    if not path.exists():
        checkpoints = Path("checkpoints") / args.model_path
        if checkpoints.exists(): path = checkpoints

    # 2. Load Model
    config = get_config(args.config)
    tokenizer = Tokenizer(str(path))
    model = Transformer(config).to(device)
    load_weights(model, str(path), device=device)
    model.eval()

    # 3. Encode Prompt
    tokens = tokenizer.encode(args.prompt, bos=True, eos=False)
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)
    stop_ids = [tokenizer.eos_id] if tokenizer.eos_id else []

    # 4. Generate Loop
    print(args.prompt, end="", flush=True) # print prompt
    
    start = time.perf_counter()
    n_tokens = 0
    
    with torch.no_grad():
        for _ in range(args.max_tokens):
            logits = model(tokens)
            next_token = sample(logits, args.temperature, args.top_k)
            
            tokens = torch.cat((tokens, next_token), dim=1)
            n_tokens += 1
            
            # Decode and print
            tid = next_token.item()
            if tid in stop_ids: break
            
            try:
                char = tokenizer.decode_token(tid)
                print(char.encode("ascii", errors="replace").decode("ascii"), end="", flush=True)
            except: pass

    # 5. Stats
    dt = time.perf_counter() - start
    tps = n_tokens / dt
    print(f"\n\nstats: {dt:.2f}s, {tps:.2f} tok/s")
    

if __name__ == "__main__":
    main()
