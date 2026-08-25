"""Engine configuration dataclass."""
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch


def to_namespace(obj):
    """Recursively wrap dicts for attribute-style access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    return obj


@dataclass(slots=True)
class Config:
    model: str
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    seed: int = 42
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    hf_config: Optional[SimpleNamespace] = None

    def __post_init__(self):
        if not os.path.isdir(self.model):
            raise ValueError(f"Model path does not exist: {self.model}")

        with open(os.path.join(self.model, "config.json"), encoding="utf-8") as f:
            self.hf_config = to_namespace(json.load(f))

        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
