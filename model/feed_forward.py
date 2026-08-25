import torch
import torch.nn as nn
import torch.nn.functional as F

from config.llama import LlamaConfig


class LlamaMLP(nn.Module):
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        hidden_dim = config.ffn_hidden_dim
        self.gate_proj = nn.Linear(config.dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
