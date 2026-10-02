import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from download_tinystories import path

torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Params
batch_size = 16
block_size = 32
lr = 3e-4
epochs = 1000
eval_interval = 20
eval_iters = 50
n_embed = 40
num_heads = 4
n_layers = 4
dropout = 0.2
max_new_tokens = 500

# load data
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# create a mapping from characters to integers
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = enc.decode

# create train / test splits
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


class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute scaled attention scores ("similarities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B,T,C) @ (B, C, T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # perform the weighted aggregation of values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,T) @ (B,T,C) = (B,T,C)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embed, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.LeakyReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embed, block_size, num_heads, dropout):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa_head = MultiHeadAttention(
            num_heads, head_size, n_embed, block_size, dropout
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
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, block_size, num_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
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

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


m = TransformerModel(vocab_size, n_embed, block_size, num_heads, n_layers, dropout)
m = m.to(device)

# print number of params
print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")

# test model
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()))

# train model
optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
for epoch in range(epochs + 1):
    xb, yb = get_batch(train_data)

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if epoch % eval_interval == 0:
        losses = estimate_loss(m, train_data, val_data, eval_iters)
        print(
            f"epoch {epoch}, train loss: {losses['train']}, val loss: {losses['val']}"
        )

print(decode(m.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()))
