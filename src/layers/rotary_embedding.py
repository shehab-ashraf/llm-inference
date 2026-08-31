"""Rotary Position Embedding (RoPE)."""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (..., D), cos/sin: (..., D/2) -> (..., D)
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (T, D/2)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # (T, D)
        self.register_buffer("cos_sin_cache", cache.unsqueeze(1), persistent=False)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: (B * S,), query: (B * S, H, D), key: (B * S, KV, D)
        # one position per token row (works for prefill & decode)
        cos_sin = self.cos_sin_cache[positions]  # (B * S, 1, D)
        cos, sin = cos_sin.chunk(2, dim=-1)  # (B * S, 1, D/2) each
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)


@lru_cache(maxsize=1)
def get_rope(head_size: int, rotary_dim: int, max_position: int, base: float) -> RotaryEmbedding:
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)
