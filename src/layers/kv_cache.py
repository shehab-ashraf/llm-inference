"""Preallocated KV cache"""

import torch


class KVCache:
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        shape = (batch_size, max_seq_len, num_kv_heads, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[layer_idx], self.v[layer_idx]
