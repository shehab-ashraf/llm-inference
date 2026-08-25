"""Sampling parameters."""
from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 1.0

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
