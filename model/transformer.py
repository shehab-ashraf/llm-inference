import torch
import torch.nn as nn
from typing import Optional

from config.llama import LlamaConfig
from model.normalization import LlamaRMSNorm
from model.attention import LlamaAttention
from model.feed_forward import LlamaMLP


class TransformerBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.dim, eps=config.norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.dim, eps=config.norm_eps)
        self.mlp = LlamaMLP(config)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self Attention
        h = x + self.self_attn(
            self.input_layernorm(x),
            position_ids,
            mask,
        )
        # Feed Forward
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = LlamaRMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
    
    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        h = self.embed_tokens(tokens)
        
        position_ids = torch.arange(start_pos, start_pos + seq_len, device=tokens.device).unsqueeze(0)
        
        for layer in self.layers:
            h = layer(h, position_ids, mask=None)
        
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits
    

