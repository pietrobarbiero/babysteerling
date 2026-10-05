import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from download_tinystories import path

torch.manual_seed(1337)

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Params
batch_size = 64
block_size = 32
lr = 3e-4
min_lr = 3e-5
warmup_steps = 200
epochs = 5000
eval_interval = 200
eval_iters = 50
n_embed = 128
num_heads = 4
n_layers = 4
dropout = 0.2
max_new_tokens = 500
weight_decay = 0.1
grad_clip = 1.0
temperature = 0.8
top_k = 50
use_amp = (
    device == "cuda"
)  # autocast/bfloat16 support on MPS is inconsistent; enable only on CUDA
use_compile = (
    device == "cuda"
)  # torch.compile on MPS is still less reliable; enable only on CUDA

# tokenizer: small custom BPE trained on this dataset (run train_tokenizer.py first)
tok = Tokenizer.from_file("./data/tinystories_tokenizer.json")
vocab_size = tok.get_vocab_size()
encode = lambda s: tok.encode(s).ids
decode = lambda ids: tok.decode(ids)

# load data
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# create train / test splits
text = text[:50_000_000]  # first ~50MB of the train split, instead of all ~1.9GB
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def get_lr(step):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, epochs - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_epochs):
    out = {}
    model.eval()
    for split, data in {"train": train_data, "val": val_data}.items():
        losses = torch.zeros(eval_epochs)
        for k in range(eval_epochs):
            x, y = get_batch(data)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class MultiHeadAttention(nn.Module):

    def __init__(
        self, num_heads, head_size, n_embed, block_size, dropout, num_kv_heads=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = (
            num_kv_heads or num_heads
        )  # e.g. num_heads=4, num_kv_heads=1 or 2
        assert num_heads % self.num_kv_heads == 0

        self.query = nn.Linear(n_embed, n_embed, bias=False)
        self.key = nn.Linear(n_embed, self.num_kv_heads * head_size, bias=False)
        self.value = nn.Linear(n_embed, self.num_kv_heads * head_size, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        q = (
            self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        )  # (B, nh, T, hs)
        k = (
            self.key(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)
        )  # (B, nkv, T, hs)
        v = (
            self.value(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)
        )  # (B, nkv, T, hs)

        # expand k,v to match num_heads by repeating each kv head across its group
        repeat_factor = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)  # (B, nh, T, hs)
        v = v.repeat_interleave(repeat_factor, dim=1)  # (B, nh, T, hs)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embed, dropout):
        super().__init__()
        hidden_dim = int(
            4 * n_embed * 2 / 3
        )  # keeps param count roughly comparable to the old 4x MLP
        self.w1 = nn.Linear(n_embed, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(n_embed, hidden_dim, bias=False)  # value
        self.w3 = nn.Linear(hidden_dim, n_embed, bias=False)  # output proj
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.silu(self.w1(x)) * self.w2(x)
        out = self.w3(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):

    def __init__(self, n_embed, block_size, num_heads, dropout):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa_head = MultiHeadAttention(
            num_heads, head_size, n_embed, block_size, dropout, num_kv_heads=2
        )
        self.ffwd = FeedForward(head_size * num_heads, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.sa_head(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size, num_heads, n_layers, dropout):
        super().__init__()
        self.block_size = block_size
        self.n_layers = n_layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, block_size, num_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # weight tying: share embedding and output projection weights
        self.lm_head.weight = self.token_embedding_table.weight

        # GPT-2 style init: small std everywhere, then scale down residual-branch
        # output projections so variance doesn't compound across layers
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w3.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T,C)
        x = token_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, C)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


m = TransformerModel(vocab_size, n_embed, block_size, num_heads, n_layers, dropout)
m = m.to(device)

if use_compile:
    m = torch.compile(m)

# print number of params
print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")

# test model before training
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(
    decode(
        m.generate(
            idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
        )[0].tolist()
    )
)

# train model
optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(epochs + 1):
    xb, yb = get_batch(train_data)

    current_lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    if use_amp:
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = m(xb, yb)
    else:
        logits, loss = m(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
    optimizer.step()

    if epoch % eval_interval == 0:
        losses = estimate_loss(m, train_data, val_data, eval_iters)
        print(
            f"epoch {epoch}, train loss: {losses['train']}, val loss: {losses['val']}, lr: {current_lr:.6f}"
        )

print(
    decode(
        m.generate(
            idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
        )[0].tolist()
    )
)
