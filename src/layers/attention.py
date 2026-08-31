"""Flash attention with KV cache support."""

import torch
from torch import nn
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from src.utils.context import get_context


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.layer_idx = layer_idx

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q: (B, S, H, D), k: (B, S, KV, D), v: (B, S, KV, D) -> (B, S, H * D)
        ctx = get_context()
        B, S = q.shape[0], q.shape[1]

        if ctx.kv_cache is None:
            out = flash_attn_func(q, k, v, causal=True, softmax_scale=self.scale)
        elif ctx.is_prefill:
            # Prefill: store K/V into cache, attend over full prompt
            k_cache, v_cache = ctx.kv_cache[self.layer_idx]
            k_cache[:, :S] = k
            v_cache[:, :S] = v
            out = flash_attn_func(q, k, v, causal=True, softmax_scale=self.scale)
        else:
            # Decode: flash_attn_with_kvcache handles cache update + attention in one kernel
            k_cache, v_cache = ctx.kv_cache[self.layer_idx]
            out = flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k,
                v,
                cache_seqlens=ctx.cache_seqlens,
                causal=True,
                softmax_scale=self.scale,
            )

        return out.reshape(B, S, self.num_heads * self.head_dim)
