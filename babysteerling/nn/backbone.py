import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    """Grouped-query self-attention: num_kv_heads < num_heads shares key/value heads across
    several query heads, for a smaller KV cache at a small quality cost.

    Causal by default. Pass attn_mask (e.g. babysteerling.diffusion's block-causal mask) to use
    a different pattern instead. This class doesn't need to know why the mask looks that way.
    """

    def __init__(self, num_heads, head_size, n_embed, dropout, num_kv_heads=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads or num_heads
        assert num_heads % self.num_kv_heads == 0

        self.query = nn.Linear(n_embed, n_embed, bias=False)
        self.key = nn.Linear(n_embed, self.num_kv_heads * head_size, bias=False)
        self.value = nn.Linear(n_embed, self.num_kv_heads * head_size, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape  # batch, sequence length, n_embed

        # project then split into per-head slices: [B, T, C] -> [B, T, heads, head_size] -> [B, heads, T, head_size]
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)

        # duplicate each kv head across the query heads that share it: [B, num_kv_heads, T, head_size] -> [B, num_heads, T, head_size]
        repeat_factor = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        if attn_mask is None:
            # is_causal=True: fused kernel, each position only sees itself and earlier ones.
            # Default path, so the common case skips the general masked kernel below.
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )  # shape: [B, num_heads, T, head_size]
        else:
            # a custom mask (e.g. block-causal) can't use the fused is_causal kernel, so it
            # goes through the general masked path instead
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )  # shape: [B, num_heads, T, head_size]

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads back: -> [B, T, C]
        out = self.proj(out)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    """SwiGLU MLP: gated activation tends to outperform plain ReLU/GELU MLPs at equal params."""

    def __init__(self, n_embed, dropout):
        super().__init__()
        hidden_dim = int(4 * n_embed * 2 / 3)  # keep param count close to a standard 4x MLP despite the extra gate projection
        self.w1 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.silu(self.w1(x)) * self.w2(x)  # shape: [B, T, hidden_dim], elementwise gate
        out = self.w3(out)  # shape: [B, T, hidden_dim] -> [B, T, n_embed]
        out = self.dropout(out)
        return out


class Block(nn.Module):
    """One pre-norm transformer block: attention sub-layer, then feedforward sub-layer."""

    def __init__(self, n_embed, num_heads, dropout, num_kv_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa_head = MultiHeadAttention(num_heads, head_size, n_embed, dropout, num_kv_heads=num_kv_heads)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, attn_mask=None):
        # normalize each sub-layer's input, not the residual stream itself
        x = x + self.sa_head(self.ln1(x), attn_mask)  # residual connection around attention
        x = x + self.ffwd(self.ln2(x))  # residual connection around the feedforward
        return x


class TokensToEmbeddings(nn.Module):
    """Token and position embeddings, without the transformer stack."""

    def __init__(self, vocab_size, n_embed, block_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)  # shape: [B, T] -> [B, T, n_embed]
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # shape: [T, n_embed]
        x = token_emb + pos_emb  # broadcast add: [B, T, n_embed] + [T, n_embed] -> [B, T, n_embed]
        return x


class TransformerModel(nn.Module):
    """Backbone only: token and position embeddings through the transformer stack.

    Stops before any LM head, since here the head applies to the bottlenecked hidden state
    (see ConceptBottleneck), not this raw output.
    """

    def __init__(self, n_embed, block_size, num_heads, n_layers, dropout, num_kv_heads, attn_mask=None):
        super().__init__()
        self.block_size = block_size
        self.n_layers = n_layers
        # ModuleList (not Sequential) so forward() can pass attn_mask through to every block
        self.blocks = nn.ModuleList([Block(n_embed, num_heads, dropout, num_kv_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)

        self.apply(self._init_weights)
        # shrink residual-writing projections so variance doesn't grow with depth (GPT-2-style init)
        for name, p in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('w3.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

        if attn_mask is not None:
            # non-persistent: it's cheap to rebuild and shouldn't be saved into/loaded from checkpoints
            self.register_buffer("_attn_mask", attn_mask, persistent=False)
        else:
            self.register_buffer("_attn_mask", None)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        for block in self.blocks:
            x = block(x, self._attn_mask)
        x = self.ln_f(x)
        return x  # shape: [B, T, n_embed]
