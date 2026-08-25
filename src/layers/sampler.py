"""Token sampler using the Gumbel-max trick."""
import torch
from torch import nn


class Sampler(nn.Module):
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        # logits: (B, V), temperatures: (B,) -> tokens: (B,)
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        return probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
