import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class ModelConfig:
    vocab_size: int = 420
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 1024
    dropout: float = 0.1

def apply_rotary_emb(x, freqs_cos, freqs_sin):
    """
    Apply Rotary Position Embedding (RoPE) to the input tensor.
    x shape: (B, seq_len, n_heads, head_dim)
    """
    # split x into real and imaginary parts (or x1, x2)
    # x: [B, T, H, D]
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    # rotate x by pi/2: (-x2, x1)
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return (x * freqs_cos) + (x_rotated * freqs_sin)
    
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)  # (end, dim/2)
    freqs_sin = torch.sin(freqs)  # (end, dim/2)
    
    # duplicate across the head_dim: (cos1, cos2..., cos1, cos2...)
    freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)
    freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)
    return freqs_cos, freqs_sin

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.wq = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wk = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wv = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size() # batch_size, seq_len, d_model
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)

        # Apply RoPE
        # q, k shape: (B, T, H, D)
        # freqs shape: (T, D) -> (1, T, 1, D)
        freqs_cos = freqs_cos.view(1, T, 1, self.head_dim)
        freqs_sin = freqs_sin.view(1, T, 1, self.head_dim)
        
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = torch.cat([cache_k, k], dim=1)
            v = torch.cat([cache_v, v], dim=1)
            
        new_kv_cache = (k, v) if kv_cache is not None or not self.training else None
            
        # Transpose for attention: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # causal mask
        is_causal = True if kv_cache is None and T > 1 else False
        
        # Flash attention using PyTorch scaled_dot_product_attention
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0.0, 
            is_causal=is_causal
        )
        
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # output projection
        y = self.resid_dropout(self.wo(y))
        return y, new_kv_cache

class SwiGLUFFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # SwiGLU: f(x) = (xW1 * sigmoid(xW1 * beta)) * xW2
        # Usually requires 3 weight matrices
        hidden_dim = config.d_ff
        # scale down hidden_dim as SwiGLU has 2 projections in the first layer
        hidden_dim = int(2 * hidden_dim / 3) 
        # make it a multiple of 256 for better memory alignment
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        # Swish(xW1) * (xW3)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = SwiGLUFFN(config)

    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        attn_out, new_cache = self.attn(self.ln_1(x), freqs_cos, freqs_sin, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache

class StyleMIDIModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # LM Head
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # tie weights
        self.tok_emb.weight = self.head.weight
        
        # init params
        self.apply(self._init_weights)
        
        # precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(self.config.d_model // self.config.n_heads, self.config.max_seq_len * 2) # * 2 to support some extrapolation
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        # discount embeddings as they are tied
        n_params -= self.tok_emb.weight.numel()
        return n_params

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        
        B, T = idx.size()
        
        # get embeddings
        x = self.tok_emb(idx)
        x = self.dropout(x)
        
        # get rope frequencies for this sequence length
        freqs_cos = self.freqs_cos[start_pos : start_pos + T]
        freqs_sin = self.freqs_sin[start_pos : start_pos + T]
        
        new_kv_caches = [] if kv_caches is not None or not self.training else None
        
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, freqs_cos, freqs_sin, cache_i)
            if new_kv_caches is not None:
                new_kv_caches.append(new_cache)
                
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            # Shift logits to match standard loss expectation, or just compute directly
            # Here logits: (B, T, V), targets: (B, T)
            # Flatten to compute CrossEntropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        return logits, loss, new_kv_caches

