"""Qwen3 model architecture."""
import torch
from torch import nn

from src.config import to_namespace
from src.layers.activation import SiluAndMul
from src.layers.attention import Attention
from src.layers.layernorm import RMSNorm
from src.layers.rotary_embedding import get_rope
from src.utils.context import get_context


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        self.rotary_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=self.scaling,
            num_kv_heads=num_kv_heads,
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # positions: (N,), hidden_states: (B, S, C)
        B, S, _ = hidden_states.shape

        # (B, S, C) -> (B * S, H, D)
        q = self.q_proj(hidden_states).view(B * S, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B * S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B * S, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        # (B * S, H, D) -> (B, H, S, D)
        q = q.view(B, S, -1, self.head_dim).transpose(1, 2)
        k = k.view(B, S, -1, self.head_dim).transpose(1, 2)
        v = v.view(B, S, -1, self.head_dim).transpose(1, 2)

        # attention -> (B, S, H * D) -> o_proj -> (B, S, C)
        out = self.attn(q, k, v)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C) -> (B, S, C)
        gate_up = torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)
        return self.down_proj(self.act_fn(gate_up))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None)
            or hidden_size // config.num_attention_heads,
            max_position=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 1_000_000.0),
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
        )
        self.mlp = Qwen3MLP(hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # hidden_states: (B, S, C)
        h = self.input_layernorm(hidden_states)
        hidden_states = hidden_states + self.self_attn(positions, h)
        h = self.post_attention_layernorm(hidden_states)
        return hidden_states + self.mlp(h)


class Qwen3Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        # input_ids: (B, S) -> hidden_states: (B, S, C)
        B, S = input_ids.shape
        if positions is None:
            positions = torch.arange(S, device=input_ids.device).repeat(B)
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(positions, hidden)
        return self.norm(hidden)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config) -> None:
        if isinstance(config, dict):
            config = to_namespace(config)
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        # input_ids: (B, S) -> hidden_states: (B, S, C)
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, S, C) -> logits: (N_rows, V)
        ctx = get_context()
        if ctx.logit_indices is not None:
            row_idx, col_idx = ctx.logit_indices
            hidden_states = hidden_states[row_idx, col_idx]
        return self.lm_head(hidden_states)
