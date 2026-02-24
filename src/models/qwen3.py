"""Qwen3 model architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Rotary embeddings

def apply_rotary_emb(x, cos, sin):
    """x: (..., D), cos/sin: (..., D/2)"""
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Precompute and cache rotary cos/sin values."""

    def __init__(self, head_size, rotary_dim, max_position_embeddings, base):
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size  # we rotate the full head
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (T, D/2)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze(1)  # (T, 1, D)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(self, positions, query, key):
        """positions: (N,), query: (N, H, D), key: (N, G, D)"""
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key   = apply_rotary_emb(key, cos, sin)
        return query, key


# -----------------------------------------------------------------------------
# Normalization

class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (no mean subtraction)."""

    def __init__(self, dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x * self.scale
        if self.bias is not None:
            x = x + self.bias
        return x.to(dtype)


# -----------------------------------------------------------------------------
# Feed-forward

class Qwen3MLP(nn.Module):
    """SwiGLU-style MLP."""

    def __init__(self, d_in, d_hidden, dtype=torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(d_in, d_hidden, bias=False, dtype=dtype)
        self.up_proj   = nn.Linear(d_in, d_hidden, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(d_hidden, d_in, bias=False, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# -----------------------------------------------------------------------------
# Attention

class Qwen3Attention(nn.Module):
    """GQA + QK RMSNorm + RoPE + SDPA."""

    def __init__(self, d_in, num_heads, head_dim, num_kv_groups,
                 qk_norm=True, dtype=torch.bfloat16, rotary_emb=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.group_size = num_heads // num_kv_groups
        self.rotary_emb = rotary_emb

        self.W_query  = nn.Linear(d_in, num_heads * head_dim, bias=False, dtype=dtype)
        self.W_key    = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value  = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(num_heads * head_dim, d_in, bias=False, dtype=dtype)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x, positions):
        """x: (B, S, C), positions: (N,) where N = B*S"""
        B, S, _ = x.shape

        # project to (N, H, D)
        q = self.W_query(x).view(B * S, self.num_heads, self.head_dim)
        k = self.W_key(x).view(B * S, self.num_kv_groups, self.head_dim)
        v = self.W_value(x).view(B * S, self.num_kv_groups, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE 
        q, k = self.rotary_emb(positions, q, k)

        # reshape for SDPA: (B, H, S, D)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # expand KV for GQA
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # attention
        ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(ctx)


# -----------------------------------------------------------------------------
# Transformer block

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb):
        super().__init__()
        hidden_size = config["hidden_size"]
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Qwen3Attention(
            d_in=hidden_size,
            num_heads=config["num_attention_heads"],
            head_dim=config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]),
            num_kv_groups=config["num_key_value_heads"],
            qk_norm=config.get("qk_norm", True),
            dtype=config["dtype"],
            rotary_emb=rotary_emb,
        )
        self.norm2 = RMSNorm(hidden_size)
        self.ff = Qwen3MLP(hidden_size, config["intermediate_size"], dtype=config["dtype"])

    def forward(self, x, positions):
        x = x + self.attn(self.norm1(x), positions)
        x = x + self.ff(self.norm2(x))
        return x


# -----------------------------------------------------------------------------
# Model

class Qwen3Model(nn.Module):
    """Qwen3 model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dtype = config["dtype"]
        hidden_size = config["hidden_size"]
        head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        self.tok_emb = nn.Embedding(config["vocab_size"], hidden_size, dtype=config["dtype"])
        self.rotary_emb = RotaryEmbedding(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=config["max_position_embeddings"],
            base=config.get("rope_theta", 1_000_000.0),
        )
        self.blocks = nn.ModuleList(
            [Qwen3DecoderLayer(config, self.rotary_emb) for _ in range(config["num_hidden_layers"])]
        )
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, config["vocab_size"], bias=False, dtype=config["dtype"])

    def forward(self, input_ids, positions=None):
        """input_ids: (B, S) token IDs, positions: (N,)"""
        B, S = input_ids.shape
        if positions is None:
            positions = torch.arange(S, device=input_ids.device)
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x, positions)
        x = self.final_norm(x)
        return self.lm_head(x.to(self.dtype))