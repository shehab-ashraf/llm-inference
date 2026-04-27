"""LLM inference engine."""

import time
import torch
import os
from typing import Any, Optional
from tokenizers import Tokenizer

from src.config import EngineConfig
from src.models.qwen3 import Qwen3Model
from src.utils.load_utils import load_weights, apply_weights, load_config
from src.sampling_params import SamplingParams
from src.sampler import Sampler


class LLM:

    def __init__(self, config: Optional[EngineConfig] = None, **kwargs):
        if config is None:
            config = EngineConfig(**kwargs)
        self.config = config

        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

        device = config.device
        dtype = config.dtype
        if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            dtype = torch.float16
            self.config.dtype = dtype

        self.tokenizer = Tokenizer.from_file(os.path.join(config.model_path, "tokenizer.json"))

        t0 = time.perf_counter()
        model_config = load_config(config.model_path)
        model_config["dtype"] = dtype
        self.model_config = model_config

        self.model = Qwen3Model(model_config)
        state_dict = load_weights(config.model_path)
        apply_weights(self.model, state_dict, model_config)
        self.model = self.model.to(device=device).eval()
        load_time = time.perf_counter() - t0

        self.sampler = Sampler()

        # print model card
        n_params = sum(p.numel() for p in self.model.parameters())
        head_dim = model_config.get("head_dim",
            model_config["hidden_size"] // model_config["num_attention_heads"])
        mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Qwen3-0.6B | {n_params/1e6:.0f}M params | {dtype} | {device}")
        print(f"  {model_config['num_hidden_layers']}L / "
              f"{model_config['num_attention_heads']}H / "
              f"{head_dim}D | loaded in {load_time:.1f}s | {mem_gb:.2f} GB VRAM")

    # -------------------------------------------------------------------------
    # Forward

    @torch.inference_mode()
    def _prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    @torch.inference_mode()
    def _decode_step(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def _sample_token(self, logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        if params.temperature == 0.0:
            return torch.argmax(logits, dim=-1)
        return self.sampler.sample(
            logits,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            min_p=params.min_p,
        )

    # -------------------------------------------------------------------------
    # Generation

    @torch.inference_mode()
    def generate(
        self,
        prompts: torch.Tensor,
        sampling_params: Optional[SamplingParams] = None,
    ) -> Any:

        if sampling_params is None:
            sampling_params = SamplingParams()

        device = self.config.device
        input_ids = prompts.to(device)

        # Prefill
        torch.cuda.synchronize()
        t_prefill_start = time.perf_counter()
        logits = self._prefill(input_ids)
        next_token = self._sample_token(logits[:, -1, :], sampling_params)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t_prefill_start

        # Decode
        torch.cuda.synchronize()
        t_decode_start = time.perf_counter()
        for step in range(1, sampling_params.max_tokens):
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            logits = self._decode_step(input_ids)  # full sequence every step, O(T^2)
            next_token = self._sample_token(logits[:, -1, :], sampling_params)
        torch.cuda.synchronize()
        tpot = (time.perf_counter() - t_decode_start) / max(1, sampling_params.max_tokens - 1)

        class GenerationResult:
            def __init__(self, output, ttft, tpot):
                self.output = output
                self.ttft = ttft
                self.tpot = tpot

        return GenerationResult(input_ids, ttft, tpot)

    def __repr__(self):
        n = sum(p.numel() for p in self.model.parameters())
        return f"LLM(model={self.config.model_path!r}, params={n/1e6:.0f}M, dtype={self.config.dtype})"