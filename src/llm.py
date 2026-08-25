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
        if dtype is None:
            cfg   = load_config(model_path).get("torch_dtype", "bfloat16")
            dtype = torch.bfloat16 if cfg == "bfloat16" else torch.float16
            if device == "cuda" and not torch.cuda.is_bf16_supported():
                dtype = torch.float16

        # Config 
        self.config = load_config(model_path)
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
        prompts: Union[str, List[str], torch.Tensor],
        sampling_params: Optional[SamplingParams] = None
    ) -> List[Dict[str, Any]]:
        """Generate from single prompt, batch of prompts, or pre-tokenized input."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Handle pre-tokenized input (token tensor)
        if isinstance(prompts, torch.Tensor):
            # prompts: (batch_size, prompt_len)
            input_ids = prompts.to(self.device)
            original_lengths = [prompts.shape[1]] * prompts.shape[0]  #  assumes fixed-length batching
        else:
            # Handle string or list of strings
            if isinstance(prompts, str):
                prompts = [prompts]

            # Tokenize all prompts
            # input_ids_list: list of tensors, each shape (1, prompt_len) - same length in batch
            input_ids_list = [self.tokenizer.encode(p, return_tensors="pt") for p in prompts]
            original_lengths = [ids.shape[1] for ids in input_ids_list]

            # Stack into batch (all prompts already same length)
            # input_ids: (batch_size, prompt_len)
            input_ids = torch.cat(input_ids_list, dim=0).to(self.device)

        batch_size = input_ids.shape[0]

        # Generate exactly max_tokens for all sequences
        for step in range(sampling_params.max_tokens):
            # Forward pass
            # logits: (batch_size, seq_len, vocab_size)
            logits = self.model(input_ids)

            # Get last token logits for each sequence in batch
            # last_logits: (batch_size, vocab_size)
            last_logits = logits[:, -1, :]

            # Sample next tokens from batch
            # next_tokens: (batch_size,)
            next_tokens = self.sampler.sample(
                last_logits,
                temperature=sampling_params.temperature,
                top_k=sampling_params.top_k,
                top_p=sampling_params.top_p,
                min_p=sampling_params.min_p
            )

            # Append to sequence
            # next_tokens_2d: (batch_size, 1)
            next_tokens_2d = next_tokens.unsqueeze(1)
            # input_ids: (batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, next_tokens_2d], dim=1)

        # Decode generated tokens
        outputs = []
        for i, original_len in enumerate(original_lengths):
            # generated_ids: [max_tokens]
            generated_ids = input_ids[i, original_len:].tolist()
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            outputs.append({"text": generated_text})

        return outputs