"""Scaled dot-product attention wrapper."""
import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    def __init__(
        self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # q: (B, H, S, D), k: (B, KV, S, D), v: (B, KV, S, D) -> (B, S, H * D)
        ctx = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=0.0,
            scale=self.scale,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        B, S = q.shape[0], q.shape[2]
        return ctx.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
