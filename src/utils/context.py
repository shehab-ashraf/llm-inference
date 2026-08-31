"""Attention metadata context singleton."""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(slots=True)
class AttnMetadata:
    is_prefill: bool = True
    logit_indices: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    kv_cache: object = None  # Optional[KVCache] - object avoids circular import
    cache_seqlens: Optional[torch.Tensor] = None


_CONTEXT = AttnMetadata()


def get_context() -> AttnMetadata:
    return _CONTEXT


def set_context(**kwargs) -> None:
    global _CONTEXT
    _CONTEXT = AttnMetadata(**kwargs)


def reset_context() -> None:
    global _CONTEXT
    _CONTEXT = AttnMetadata()
