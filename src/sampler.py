"""Token sampling via FlashInfer LogitsPipe."""

import torch
from flashinfer.logits_processor import (
    LogitsPipe, Temperature, Softmax, TopK, TopP, MinP, Sample
)
from typing import Optional


class Sampler:

    def __init__(self):
        self.pipe = LogitsPipe([
            Temperature(), Softmax(), TopK(), TopP(), MinP(), Sample()
        ])

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
    ) -> torch.Tensor:
        """logits: (B, V) -> next_token: (B, 1)"""
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1)
        return self.pipe(logits, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)
