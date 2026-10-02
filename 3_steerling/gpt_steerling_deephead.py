import bisect
import json
import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

torch.manual_seed(1337)

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Params
batch_size = 64
block_size = 256
lr = 3e-4
min_lr = 3e-5
warmup_steps = 200
max_steps = 5000
eval_interval = 200
eval_iters = 50
n_embed = 128
num_heads = 4
num_kv_heads = 2
n_layers = 4
dropout = 0.2
max_new_tokens = 500
weight_decay = 0.1
grad_clip = 1.0
temperature = 0.8
top_k = 50

# Concept bottleneck params
unknown_ratio = 3  # m = unknown_ratio * n known concepts
unknown_rank = (
    None  # set an int (e.g. 32) to low-rank factorize the unknown embedding table
)
top_k_known = None  # set an int to sparsify known-concept activations per token
top_k_unknown = None  # set an int to sparsify unknown-concept activations per token
p_epsilon = 0.1  # residual dropout
lambda_concept = 1.0
lambda_rec = 1.0
lambda_indep = 1.0

ckpt_dir = "./model"
ckpt_path = os.path.join(ckpt_dir, "steerling_deephead.pt")

# data: reuse 3_atlas's concept-annotated TinyStories (tokens + per-document concept labels)
data_dir = "../2_atlas/data"
tokens_path = os.path.join(data_dir, "steerling_tokens.pt")
concepts_path = os.path.join(data_dir, "steerling_concepts.pt")
concept_lib_path = os.path.join(data_dir, "concepts.json")
tokenizer_path = os.path.join(data_dir, "tinystories_tokenizer.json")

tok = Tokenizer.from_file(tokenizer_path)
vocab_size = tok.get_vocab_size()
decode = lambda ids: tok.decode(ids)

data = torch.load(tokens_path)
doc_records = torch.load(concepts_path)  # [{chunk_id, start, end, concept_ids}, ...]
with open(concept_lib_path) as f:
    concept_library = json.load(f)
n_concepts = len(concept_library)
print(
    f"Loaded {len(data)} tokens, {len(doc_records)} documents, {n_concepts} known concepts"
)

n = int(0.9 * len(data))
# documents are laid out contiguously and non-overlapping, so a sorted list of starts lets us
# binary-search the small contiguous range overlapping any sampled window
doc_starts = [d["start"] for d in doc_records]


def overlapping_docs(window_start, window_end):
    i = max(bisect.bisect_right(doc_starts, window_start) - 1, 0)
    spans = []
    while i < len(doc_records) and doc_records[i]["start"] < window_end:
        d = doc_records[i]
        s, e = max(d["start"], window_start), min(d["end"], window_end)
        if e > s:
            spans.append((s - window_start, e - window_start, d["concept_ids"]))
        i += 1
    return spans


def build_supervision(starts):
    # per-window (batch_idx, tok_start, tok_end, concept_ids) tuples for ConceptLoss, and a
    # dense per-token multi-hot label tensor for the bottleneck's reconstruction-loss target
    doc_spans = []
    known_labels = torch.zeros(len(starts), block_size, n_concepts, device=device)
    for b, ws in enumerate(starts):
        for s, e, concept_ids in overlapping_docs(ws, ws + block_size):
            doc_spans.append((b, s, e, concept_ids))
            known_labels[b, s:e, concept_ids] = 1.0
    return doc_spans, known_labels


def get_batch(split):
    lo, hi = (0, n) if split == "train" else (n, len(data))
    ix = torch.randint(lo, hi - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y, ix.tolist()


def get_lr(step):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss():
    out = {}
    backbone.eval()
    bottleneck.eval()
    head.eval()
    for split in ("train", "val"):
        totals = {"total": 0.0, "lm": 0.0, "concept": 0.0, "rec": 0.0, "indep": 0.0}
        for _ in range(eval_iters):
            xb, yb, starts = get_batch(split)
            loss, components = compute_loss(xb, yb, starts)
            totals["total"] += loss.item()
            for key in ("lm", "concept", "rec", "indep"):
                totals[key] += components[key]
        out[split] = {key: value / eval_iters for key, value in totals.items()}
    backbone.train()
    bottleneck.train()
    head.train()
    return out


class MultiHeadAttention(nn.Module):

    def __init__(
        self, num_heads, head_size, n_embed, block_size, dropout, num_kv_heads=None
    ):
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

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)

        repeat_factor = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

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
        hidden_dim = int(4 * n_embed * 2 / 3)
        self.w1 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.silu(self.w1(x)) * self.w2(x)
        out = self.w3(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):

    def __init__(self, n_embed, block_size, num_heads, dropout, num_kv_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa_head = MultiHeadAttention(
            num_heads,
            head_size,
            n_embed,
            block_size,
            dropout,
            num_kv_heads=num_kv_heads,
        )
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.sa_head(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x


class TransformerModel(nn.Module):
    """Backbone only: token/position embeddings through the transformer stack. Stops before
    any LM head -- the head applies to the bottlenecked hidden state, not this raw output.
    """

    def __init__(
        self,
        vocab_size,
        n_embed,
        block_size,
        num_heads,
        n_layers,
        dropout,
        num_kv_heads,
    ):
        super().__init__()
        self.block_size = block_size
        self.n_layers = n_layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[
                Block(n_embed, block_size, num_heads, dropout, num_kv_heads)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embed)

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

    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return x


def sparsify_top_k(activations, k):
    # keep only the top-k largest activations per token, zero the rest (optional, off by default)
    if k is None or k >= activations.shape[-1]:
        return activations
    top_vals, top_idx = torch.topk(activations, k, dim=-1)
    sparse = torch.zeros_like(activations)
    sparse.scatter_(-1, top_idx, top_vals)
    return sparse


class SupervisedConceptHead(nn.Module):
    """Known concepts (size n, fixed by the concept library)."""

    def __init__(self, d, n, hidden_dim=None, top_k=None):
        super().__init__()
        hidden_dim = hidden_dim or d
        self.f = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n)
        )
        self.K = nn.Parameter(torch.randn(n, d) * 0.02)  # known concept embedding table
        self.top_k = top_k

    def forward(self, h):
        k = torch.sigmoid(self.f(h))
        k = sparsify_top_k(k, self.top_k)
        k_hat = k @ self.K
        return k, k_hat

    def ground_truth_embedding(self, known_labels):
        # weighted sum of K by the ground-truth chunk-level labels, broadcast to every token
        # position in the chunk -- this is the reconstruction loss's target for the unknown head
        return known_labels.float() @ self.K


class UnsupervisedConceptHead(nn.Module):
    """Unknown concepts (size m = unknown_ratio * n), with an optional low-rank embedding."""

    def __init__(self, d, m, hidden_dim=None, rank=None, top_k=None):
        super().__init__()
        hidden_dim = hidden_dim or d
        self.g = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, m)
        )
        self.rank = rank
        if rank is None:
            self.U = nn.Parameter(torch.randn(m, d) * 0.02)
        else:
            # U = A @ B factorization, cheaper when m is large relative to d
            self.A = nn.Parameter(torch.randn(m, rank) * 0.02)
            self.B = nn.Parameter(torch.randn(rank, d) * 0.02)
        self.top_k = top_k

    def _embed(self, u):
        if self.rank is None:
            return u @ self.U
        return (u @ self.A) @ self.B

    def forward(self, h):
        u = torch.sigmoid(self.g(h))
        u = sparsify_top_k(u, self.top_k)
        u_hat = self._embed(u)
        return u, u_hat


class ResidualModule(nn.Module):
    """epsilon = h - k_hat - u_hat, with dropout to discourage relying on the residual."""

    def __init__(self, p_epsilon=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p_epsilon)

    def forward(self, h, k_hat, u_hat):
        epsilon = h - k_hat - u_hat
        return self.dropout(epsilon)


class ConceptBottleneck(nn.Module):
    """Composes the three heads: h_bar = k_hat + u_hat + epsilon."""

    def __init__(
        self,
        d,
        n,
        unknown_ratio=3,
        p_epsilon=0.1,
        unknown_rank=None,
        top_k_known=None,
        top_k_unknown=None,
    ):
        super().__init__()
        self.n = n
        self.m = unknown_ratio * n
        self.known = SupervisedConceptHead(d, n, top_k=top_k_known)
        self.unknown = UnsupervisedConceptHead(
            d, self.m, rank=unknown_rank, top_k=top_k_unknown
        )
        self.residual = ResidualModule(p_epsilon)

    def forward(self, h, known_labels=None):
        k, k_hat = self.known(h)
        u, u_hat = self.unknown(h)

        k_hat_gt, u_hat_gt = None, None
        if known_labels is not None:
            k_hat_gt = self.known.ground_truth_embedding(known_labels)
            u_hat_gt = h - k_hat_gt

        epsilon = self.residual(h, k_hat, u_hat)
        h_bar = k_hat + u_hat + epsilon

        intermediates = {
            "k": k,
            "u": u,
            "k_hat": k_hat,
            "u_hat": u_hat,
            "k_hat_gt": k_hat_gt,
            "u_hat_gt": u_hat_gt,
            "epsilon": epsilon,
        }
        return h_bar, intermediates


class ConceptLoss(nn.Module):
    """OR-aggregated BCE over per-document concept labels."""

    def forward(self, k, doc_spans):
        if not doc_spans:
            return torch.tensor(0.0, device=k.device)
        n = k.shape[-1]
        total = k.new_zeros(())
        for batch_idx, tok_start, tok_end, concept_ids in doc_spans:
            k_span = k[batch_idx, tok_start:tok_end, :]
            k_chunk = 1 - torch.prod(1 - k_span, dim=0)  # OR-aggregation over the chunk
            y = k.new_zeros(n)
            y[concept_ids] = 1.0
            total = total + F.binary_cross_entropy(
                k_chunk.clamp(1e-6, 1 - 1e-6), y, reduction="sum"
            )
        return total / len(doc_spans)


class ReconstructionLoss(nn.Module):
    """MSE between the unknown head's u_hat and its ground-truth target."""

    def forward(self, u_hat, u_hat_gt):
        if u_hat_gt is None:
            return torch.tensor(0.0, device=u_hat.device)
        return ((u_hat - u_hat_gt) ** 2).mean()


class IndependenceLoss(nn.Module):
    """Cross-covariance penalty between k_hat and u_hat; gradients flow only through u_hat."""

    def forward(self, k_hat, u_hat):
        d = k_hat.shape[-1]
        Hk = k_hat.detach().reshape(-1, d)
        Hu = u_hat.reshape(-1, d)
        num_tokens = Hk.shape[0]

        Phi = Hk - Hk.mean(dim=0, keepdim=True)
        Psi = Hu - Hu.mean(dim=0, keepdim=True)
        cross_cov = Psi.t() @ Phi
        return (cross_cov**2).sum() / (d**2 * max(num_tokens - 1, 1))


class ConceptLMHead(nn.Module):
    """No bias, so logits are an exact linear function of h_bar: decompose() sums to forward()."""

    def __init__(self, d, vocab_size, tied_embedding=None):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d, d, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d, vocab_size, bias=False),
        )
        # if tied_embedding is not None:
        #     self.head.weight = tied_embedding

    def forward(self, h_bar):
        return self.head(h_bar)

    def decompose(self, k_hat, u_hat, epsilon):
        return self.head(k_hat), self.head(u_hat), self.head(epsilon)


backbone = TransformerModel(
    vocab_size, n_embed, block_size, num_heads, n_layers, dropout, num_kv_heads
).to(device)
bottleneck = ConceptBottleneck(
    n_embed,
    n_concepts,
    unknown_ratio=unknown_ratio,
    p_epsilon=p_epsilon,
    unknown_rank=unknown_rank,
    top_k_known=top_k_known,
    top_k_unknown=top_k_unknown,
).to(device)
head = ConceptLMHead(
    n_embed, vocab_size, tied_embedding=backbone.token_embedding_table.weight
).to(device)

concept_loss_fn = ConceptLoss()
rec_loss_fn = ReconstructionLoss()
indep_loss_fn = IndependenceLoss()

# dedupe: head.weight is tied to backbone's token embedding, so naively concatenating each
# module's .parameters() would list that tensor twice and double-step it in the optimizer
all_params = list(
    dict.fromkeys(
        list(backbone.parameters())
        + list(bottleneck.parameters())
        + list(head.parameters())
    )
)
print(sum(p.numel() for p in all_params) / 1e6, "M params")


def compute_loss(xb, yb, starts):
    doc_spans, known_labels = build_supervision(starts)
    h = backbone(xb)
    h_bar, intermediates = bottleneck(h, known_labels=known_labels)
    logits = head(h_bar)

    B, T, C = logits.shape
    lm_loss = F.cross_entropy(logits.view(B * T, C), yb.view(B * T))
    concept_loss = concept_loss_fn(intermediates["k"], doc_spans)
    rec_loss = rec_loss_fn(intermediates["u_hat"], intermediates["u_hat_gt"])
    indep_loss = indep_loss_fn(intermediates["k_hat"], intermediates["u_hat"])

    total_loss = (
        lm_loss
        + lambda_concept * concept_loss
        + lambda_rec * rec_loss
        + lambda_indep * indep_loss
    )
    components = {
        "lm": lm_loss.item(),
        "concept": concept_loss.item(),
        "rec": rec_loss.item(),
        "indep": indep_loss.item(),
    }
    return total_loss, components


if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone"])
    bottleneck.load_state_dict(ckpt["bottleneck"])
    head.load_state_dict(ckpt["head"])
    print(f"Loaded existing checkpoint from {ckpt_path}, skipping training.")
else:
    print("No existing checkpoint found, training from scratch.")

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    for step in range(max_steps + 1):
        xb, yb, starts = get_batch("train")

        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        loss, _ = compute_loss(xb, yb, starts)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()

        if step % eval_interval == 0:
            losses = estimate_loss()

            def fmt(split):
                s = losses[split]
                return (
                    f"{s['total']:.4f} (lm {s['lm']:.4f} concept {s['concept']:.4f} "
                    f"rec {s['rec']:.4f} indep {s['indep']:.4f})"
                )

            print(
                f"step {step}, train loss: {fmt('train')}, val loss: {fmt('val')}, lr: {current_lr:.6f}"
            )

    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {
            "backbone": backbone.state_dict(),
            "bottleneck": bottleneck.state_dict(),
            "head": head.state_dict(),
        },
        ckpt_path,
    )
    print(f"Training complete, saved checkpoint to {ckpt_path}")


@torch.no_grad()
def generate(idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        h = backbone(idx_cond)
        h_bar, _ = bottleneck(h)
        logits = head(h_bar)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# generate a sample
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(
    decode(
        generate(
            idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
        )[0].tolist()
    )
)
