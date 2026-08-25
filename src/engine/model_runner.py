"""GPU execution unit: manages model and sampler forward passes."""
import torch

from src.config import Config
from src.layers.sampler import Sampler
from src.models.qwen3 import Qwen3ForCausalLM
from src.utils.context import reset_context, set_context
from src.utils.loader import load_model

_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class ModelRunner:
    def __init__(self, config: Config) -> None:
        self.config = config
        hf_config = config.hf_config

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(
            _TORCH_DTYPES.get(
                getattr(hf_config, "torch_dtype", "bfloat16"), torch.bfloat16
            )
        )
        try:
            self.model = Qwen3ForCausalLM(hf_config)
            load_model(self.model, config.model)
        finally:
            torch.set_default_dtype(default_dtype)

        self.model = self.model.to(config.device).eval()
        self.sampler = Sampler().to(config.device)

    @torch.inference_mode()
    def run(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logit_indices: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # input_ids: (B, S), positions: (N,) -> logits: (B, V)
        set_context(is_prefill=True, logit_indices=logit_indices)
        try:
            hidden = self.model(input_ids, positions)
            return self.model.compute_logits(hidden)
        finally:
            reset_context()

    def sample(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        # logits: (B, V), temperatures: (B,) -> tokens: (B,)
        return self.sampler(logits, temperatures)
