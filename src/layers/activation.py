"""SwiGLU activation function."""
import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 2 * d) -> (..., d)
        x, y = x.chunk(2, dim=-1)
        return F.silu(x) * y
