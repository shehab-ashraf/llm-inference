from dataclasses import dataclass
from typing import Optional


@dataclass
class LlamaConfig:
    
    # Model dimensions
    dim: int                      # Hidden dimension (d_model)
    n_layers: int                 # Number of transformer blocks
    n_heads: int                  # Number of attention heads (for Q)
    n_kv_heads: Optional[int]     # Number of KV heads (for GQA). None = same as n_heads (MHA)
    
    # Vocabulary
    vocab_size: int               # Size of tokenizer vocabulary
    
    # FFN dimensions
    # Hidden dim formula: int(2/3 * 4 * dim), rounded to multiple of `multiple_of`
    multiple_of: int = 256        # FFN hidden dim must be multiple of this
    ffn_dim_multiplier: Optional[float] = None  # Custom multiplier (Llama-3 uses this)
    
    # Normalization
    norm_eps: float = 1e-5        # RMSNorm epsilon
    
    # Position embeddings (RoPE)
    rope_theta: float = 10000.0   # Base frequency for RoPE
    max_seq_len: int = 2048       # Maximum sequence length
    
    # Computation
    dtype: str = "float32"        # Default dtype: float32, float16, bfloat16
    
    def __post_init__(self):

        # If n_kv_heads not specified, use MHA (n_kv_heads = n_heads)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
    
    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads
    
    @property
    def ffn_hidden_dim(self) -> int:

        if self.ffn_dim_multiplier is not None:
            hidden_dim = int(self.ffn_dim_multiplier * self.dim)
        else:
            hidden_dim = int(2 * self.dim * 4 / 3)
        
        # Round up to multiple of `multiple_of`
        hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)
        return hidden_dim

MODEL_PARAMS = {
    # TinyStories
    "stories-15m":    dict(dim=288,  n_layers=6,  n_heads=6,  n_kv_heads=6,  vocab_size=32000, max_seq_len=256),
    "stories-42m":    dict(dim=512,  n_layers=8,  n_heads=8,  n_kv_heads=8,  vocab_size=32000, max_seq_len=1024),
    "stories-110m":   dict(dim=768,  n_layers=12, n_heads=12, n_kv_heads=12, vocab_size=32000, max_seq_len=1024),
    "tinyllama-1.1b": dict(dim=2048, n_layers=22, n_heads=32, n_kv_heads=4,  vocab_size=32000, max_seq_len=2048),

    # Llama-2
    "llama2-7b":      dict(dim=4096, n_layers=32, n_heads=32, n_kv_heads=32, vocab_size=32000, max_seq_len=4096),
    "llama2-13b":     dict(dim=5120, n_layers=40, n_heads=40, n_kv_heads=40, vocab_size=32000, max_seq_len=4096),
    "llama2-70b":     dict(dim=8192, n_layers=80, n_heads=64, n_kv_heads=8,  vocab_size=32000, max_seq_len=4096),

    # CodeLlama
    "codellama-7b":   dict(dim=4096, n_layers=32, n_heads=32, n_kv_heads=32, vocab_size=32016, max_seq_len=16384, rope_theta=1e6),
    "codellama-34b":  dict(dim=8192, n_layers=48, n_heads=64, n_kv_heads=8,  vocab_size=32016, max_seq_len=16384, rope_theta=1e6),

    # Llama-3 (8B, 70B)
    "llama3-8b":      dict(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8,  vocab_size=128256, max_seq_len=8192, rope_theta=5e5, ffn_dim_multiplier=1.3),
    "llama3-70b":     dict(dim=8192, n_layers=80, n_heads=64, n_kv_heads=8,  vocab_size=128256, max_seq_len=8192, rope_theta=5e5, ffn_dim_multiplier=1.3),

    # OpenLLaMA
    "open-llama-7b":  dict(dim=4096, n_layers=32, n_heads=32, n_kv_heads=32, vocab_size=32000, max_seq_len=2048, norm_eps=1e-6),
}

def get_config(name: str) -> LlamaConfig:
    if name not in MODEL_PARAMS:
        raise ValueError(f"Unknown config: {name}. Available: {list(MODEL_PARAMS.keys())}")
    
    return LlamaConfig(**MODEL_PARAMS[name])
