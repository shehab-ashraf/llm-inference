"""Token sampling utilities using FlashInfer LogitsPipe."""

import torch
from flashinfer.logits_processor import (
    LogitsPipe,
    Temperature,
    Softmax,
    TopK,
    TopP,
    MinP,
    Sample
)
from typing import Optional



class Sampler:

    def __init__(self):
        """Initialize LogitsPipe."""
        self.pipe = LogitsPipe([
            Temperature(),   # Apply temperature
            Softmax(),       # Logits → probabilities
            TopK(),          # Top-k filtering
            TopP(),          # Nucleus (top-p) filtering
            MinP(),          # Modern min-p filter (recommended)
            Sample()         # Sample token
        ])

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample next token from logits.

        Args:
            logits: [batch_size, vocab_size]
            temperature: Sampling temperature
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold
            min_p: Min probability threshold

        Returns:
            next_token: [batch_size, 1] sampled token IDs
        """
        # Greedy sampling
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1)

        kwargs = {"temperature": temperature}
        kwargs["top_k"] = top_k
        kwargs["top_p"] = top_p
        kwargs["min_p"] = min_p

        return self.pipe(logits, **kwargs)
