"""LLM interface."""
import time
import torch
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Union

from src.models.qwen3 import Qwen3Model
from src.utils.load_utils import load_weights, apply_weights, load_config
from src.sampling_params import SamplingParams
from src.sampler import Sampler


class LLM:
    """Main LLM interface."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize LLM.

        Args:
            model_path: Path to model directory
            device: 'cuda' or 'cpu'
            dtype: Model dtype
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Config from file (single source: head_dim, torch_dtype, etc. live in config.json)
        self.config = load_config(model_path)
        # dtype: use config's torch_dtype when not specified, else respect device (bf16 if supported)
        if dtype is None:
            cfg_dtype = self.config.get("torch_dtype", "bfloat16")
            if isinstance(cfg_dtype, str):
                dtype = torch.bfloat16 if cfg_dtype == "bfloat16" else torch.float16
            else:
                dtype = cfg_dtype
            if device == "cuda" and not torch.cuda.is_bf16_supported() and dtype == torch.bfloat16:
                dtype = torch.float16
        self.config["dtype"] = dtype

        self.model_path = model_path
        self.device = device
        self.dtype = dtype

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model + weights
        t0 = time.perf_counter()
        self.model = Qwen3Model(self.config)
        state_dict = load_weights(model_path)
        apply_weights(self.model, state_dict, self.config)
        self.model = self.model.to(device=device)
        self.model.eval()
        load_time = time.perf_counter() - t0

        # Sampler
        self.sampler = Sampler()

        # Log (head_dim is in config.json for Qwen3; others fallback to hidden_size // num_heads)
        n_params = sum(p.numel() for p in self.model.parameters())
        head_dim = self.config.get("head_dim", self.config["hidden_size"] // self.config["num_attention_heads"])
        print(f"Qwen3-0.6B | {n_params/1e6:.0f}M params | {dtype} | {device}")
        print(f"  {self.config['num_hidden_layers']}L / {self.config['num_attention_heads']}H / "
              f"{head_dim}D | loaded in {load_time:.1f}s")

    @torch.inference_mode()
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None
    ) -> List[Dict[str, Any]]:
        """prompts: Single prompt or list of prompts, sampling_params: Sampling configuration"""
        if isinstance(prompts, str):
            prompts = [prompts]

        if sampling_params is None:
            sampling_params = SamplingParams()

        outputs = []
        for prompt in prompts:
            output = self._generate_single(prompt, sampling_params)
            outputs.append(output)

        return outputs

    def _generate_single(
        self,
        prompt: str,
        params: SamplingParams
    ) -> Dict[str, Any]:
        """prompt: Single prompt, params: Sampling configuration"""
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        original_length = input_ids.shape[1]

        # Generate tokens one by one
        for _ in range(params.max_tokens):
            # Full forward pass every step
            logits = self.model(input_ids)  # (batch_size, S, vocab_size)

            # Last token logits (keep batch dim so sampler returns [batch_size])
            last_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Sample next token -> (batch_size,) then unsqueeze(1) -> (batch_size, 1) for concatenation
            next_token = self.sampler.sample(
                last_logits,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                min_p=params.min_p
            )
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

            # Stopping condition (batch_size=1: next_token is (1,) → .item() gives Python int)
            if not params.ignore_eos:
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        # Decode generated tokens
        generated_ids = input_ids[0, original_length:].tolist()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {"text": generated_text}
