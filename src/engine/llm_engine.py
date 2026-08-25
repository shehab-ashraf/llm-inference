"""LLM engine orchestrator: manages tokenizer, ModelRunner, and generation loop."""
import os
import time
from dataclasses import dataclass, fields

import torch
from tokenizers import Tokenizer

from src.config import Config
from src.engine.model_runner import ModelRunner
from src.sampling_params import SamplingParams


@dataclass(slots=True)
class GenerationResult:
    output: torch.Tensor
    ttft: float
    tpot: float


class LLMEngine:
    def __init__(self, model_path: str, **kwargs):
        valid = {f.name for f in fields(Config)}
        self.config = Config(
            model=model_path, **{k: v for k, v in kwargs.items() if k in valid}
        )

        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        if (
            self.config.device == "cuda"
            and self.config.dtype == torch.bfloat16
            and not torch.cuda.is_bf16_supported()
        ):
            self.config.dtype = torch.float16

        self.tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))

        t0 = time.perf_counter()
        self.model_runner = ModelRunner(self.config)
        load_time = time.perf_counter() - t0

        self._print_model_card(load_time)

    def _print_model_card(self, load_time: float) -> None:
        cfg, hf = self.config, self.config.hf_config
        n_params = sum(p.numel() for p in self.model_runner.model.parameters())
        head_dim = (
            getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        )
        mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        name = os.path.basename(os.path.normpath(cfg.model))
        print(f"{name} | {n_params/1e6:.0f}M params | {cfg.dtype} | {cfg.device}")
        print(
            f"  {hf.num_hidden_layers}L / {hf.num_attention_heads}H / "
            f"{head_dim}D | loaded in {load_time:.1f}s | {mem_gb:.2f} GB VRAM"
        )

    def _logit_indices_and_positions(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # input_ids: (B, S) -> positions: (B * S,), logit_indices: (B,), (B,)
        B, S = input_ids.shape
        logit_indices = (
            torch.arange(B, device=input_ids.device),
            torch.full((B,), S - 1, device=input_ids.device),
        )
        positions = torch.arange(S, device=input_ids.device).repeat(B)
        return positions, logit_indices

    @torch.inference_mode()
    def _prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, S) -> logits: (B, V)
        positions, idx = self._logit_indices_and_positions(input_ids)
        return self.model_runner.run(input_ids, positions, idx)

    @torch.inference_mode()
    def _decode_step(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, S_curr) -> logits: (B, V)
        positions, idx = self._logit_indices_and_positions(input_ids)
        return self.model_runner.run(input_ids, positions, idx)

    def _sample_token(
        self, logits: torch.Tensor, params: SamplingParams
    ) -> torch.Tensor:
        # logits: (B, V) -> token_ids: (B,)
        if params.temperature == 0.0:
            return logits.argmax(dim=-1)
        temps = torch.full(
            (logits.size(0),),
            params.temperature,
            device=logits.device,
            dtype=torch.float32,
        )
        return self.model_runner.sample(logits, temps)

    @torch.inference_mode()
    def generate(
        self, prompts: torch.Tensor, sampling_params: SamplingParams | None = None
    ) -> GenerationResult:
        if sampling_params is None:
            sampling_params = SamplingParams()

        input_ids = prompts.to(self.config.device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = self._prefill(input_ids)
        next_token = self._sample_token(logits, sampling_params)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(1, sampling_params.max_tokens):
            input_ids = torch.cat((input_ids, next_token.unsqueeze(1)), dim=1)
            logits = self._decode_step(input_ids)
            next_token = self._sample_token(logits, sampling_params)
        torch.cuda.synchronize()
        tpot = (time.perf_counter() - t0) / max(1, sampling_params.max_tokens - 1)

        return GenerationResult(input_ids, ttft, tpot)
