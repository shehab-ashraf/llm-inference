"""Engine configuration."""

from dataclasses import dataclass
import torch


@dataclass
class EngineConfig:
    model_path: str = "./Qwen3-0.6B"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    max_model_len: int = 4096
    seed: int = 42
    gpu_memory_utilization: float = 0.9
