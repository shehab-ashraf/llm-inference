"""Sampling parameters for text generation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingParams:
    """
    Parameters for text generation.

    Args:
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_k: Keep only top k tokens
        top_p: Nucleus sampling threshold
        min_p: Min probability threshold
        ignore_eos: Continue after EOS token
    """
    max_tokens: int = 64
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    ignore_eos: bool = False

    def __post_init__(self):
        """Validate parameters."""
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {self.top_k}")
        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.min_p is not None and not (0 < self.min_p <= 1):
            raise ValueError(f"min_p must be in (0, 1], got {self.min_p}")
