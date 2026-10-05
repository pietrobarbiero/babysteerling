"""Steerling: Causal Diffusion backbone + concept bottleneck, as one flat script.

Same style as 0_gpt_chars/gpt.py and 1_gpt_tokens/gpt_light.py -- a single, top-to-bottom
script instead of a package -- reorganized from what used to be a multi-file module
(diffusion.py / backbone.py / concept_bottleneck.py / classification_head.py / losses.py /
model.py / train.py) back into one file, since none of those pieces are reused anywhere else.

High-level idea (paper: Guide Labs, "Scaling Inherently Interpretable Language Models"):
  1. Backbone: instead of a standard causal transformer, we use "Causal Diffusion" -- attention
     is bidirectional *within* small blocks of tokens and causal *across* blocks -- combined with
     a masked-diffusion training objective (predict randomly masked tokens, not just the next
     one). This gives the model a trained notion of "no information here" ([MASK]), which a
     plain next-token model never learns, and which later enables faithful input attribution.
  2. Concept bottleneck: the backbone's hidden state is decomposed into three additive pieces --
     known concepts (supervised by the concept library from ../3_atlas), unknown concepts (free
     capacity the model discovers on its own), and a residual -- before the final projection to
     vocabulary logits. Because that projection is linear, every output logit becomes an exact
     sum of a known-concept contribution, an unknown-concept contribution, and a residual
     contribution: the model's predictions are attributable back to specific concepts by
     construction, not via a post-hoc probe.
  3. Four losses train this: the language modeling loss (did we predict the masked tokens
     correctly?), the concept loss (did the known-concept head predict the right concepts?), the
     reconstruction loss (does the unknown head capture what the known concepts don't?), and the
     independence loss (are known and unknown kept from encoding the same thing redundantly?).

This version follows the paper's architecture closely; see ../5_steerlingv2/gpt_steerling.py for
a simpler variant that drops the diffusion objective and teacher forcing in favor of a plain
autoregressive backbone with the same concept bottleneck and losses.
"""

import bisect
import json
import math
import os

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.nn import functional as F

torch.manual_seed(1337)

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Params
seq_len = 128  # context length (tokens per training window)
diff_block_len = (
    32  # diffusion block size b: attention is bidirectional within a block of
)
# this many tokens and causal across blocks (Section 5.2)
n_embed = 128
num_heads = 4
num_kv_heads = 2
n_layers = 4
dropout = 0.2

# concept bottleneck
unknown_ratio = 3  # m (unknown concepts) = unknown_ratio * n (known concepts)
unknown_rank = (
    32  # low-rank factorization of the unknown embedding table (U = A @ B); set to
)
# None for a dense table instead
p_cfg = 0.1  # classifier-free-guidance-style dropout on the known head's activations
p_epsilon = 0.1  # residual dropout

# loss weights (paper Eq. 15)
lambda_concept = 1.0
lambda_rec = 1.0
lambda_indep = 1.0

batch_size = 32
lr = 3e-4
min_lr = 3e-5
warmup_steps = 200
max_steps = 3000
eval_interval = 200
eval_iters = 20
grad_clip = 1.0
weight_decay = 0.1

gen_steps = 32  # number of denoising steps used by the sanity-check sampler at the end
gen_temperature = 0.8
gen_top_k = 50

data_dir = "../2_atlas/data"
tokens_path = os.path.join(data_dir, "steerling_tokens.pt")
concepts_path = os.path.join(data_dir, "steerling_concepts.pt")
concept_lib_path = os.path.join(data_dir, "concepts.json")
tokenizer_path = os.path.join(data_dir, "tinystories_tokenizer.json")

ckpt_dir = "./model"
ckpt_path = os.path.join(ckpt_dir, "steerling.pt")

# tokenizer: reuse 3_atlas's BPE tokenizer, adding [MASK] as one extra vocab entry (existing
# token ids are untouched, so steerling_tokens.pt doesn't need re-tokenizing). [MASK] is the
# "trained absence baseline" the diffusion objective needs -- see the module docstring.
tok = Tokenizer.from_file(tokenizer_path)
if tok.token_to_id("[MASK]") is None:
    tok.add_special_tokens(["[MASK]"])
mask_token_id = tok.token_to_id("[MASK]")
vocab_size = tok.get_vocab_size()
decode = lambda ids: tok.decode(ids)

# data: token stream + per-document concept labels produced by ../3_atlas's pipeline
tokens = torch.load(tokens_path)  # shape: [N_total_tokens], whole corpus concatenated
doc_records = torch.load(
    concepts_path
)  # [{chunk_id, start, end, concept_ids}, ...], one per document
with open(concept_lib_path) as f:
    concept_library = json.load(f)
n_concepts = len(concept_library)
print(
    f"Loaded {len(tokens)} tokens, {len(doc_records)} documents, {n_concepts} known concepts"
)

n_split = int(0.9 * len(tokens))
# documents are laid out contiguously and non-overlapping (each one's end equals the next one's
# start), so a sorted list of starts lets us binary-search straight to the short contiguous run
# overlapping any sampled window, instead of scanning every document for every batch
doc_starts = [d["start"] for d in doc_records]


def overlapping_docs(window_start, window_end):
    """Find every document whose token span intersects [window_start, window_end).
    Returns (local_start, local_end, concept_ids) tuples, offsets relative to window_start.
    """
    i = max(
        bisect.bisect_right(doc_starts, window_start) - 1, 0
    )  # first candidate document
    spans = []
    while i < len(doc_records) and doc_records[i]["start"] < window_end:
        d = doc_records[i]
        s, e = max(d["start"], window_start), min(
            d["end"], window_end
        )  # intersect with window
        if e > s:
            spans.append(
                (s - window_start, e - window_start, d["concept_ids"])
            )  # -> window-local offsets
        i += 1
    return spans


def get_batch(split):
    """Sample a batch of random contiguous token windows for `split` ('train' or 'val')."""
    lo, hi = (0, n_split) if split == "train" else (n_split, len(tokens))
    starts = torch.randint(
        lo, hi - seq_len, (batch_size,)
    )  # shape: [batch_size], window start offsets
    x = torch.stack(
        [tokens[s : s + seq_len] for s in starts]
    )  # shape: [batch_size, seq_len]
    return x.to(device), starts.tolist()


def build_supervision(starts):
    """Build the concept-loss/reconstruction-loss supervision for a batch of windows.

    doc_spans: (batch_idx, tok_start, tok_end, concept_ids) per document overlapping any window
        in the batch -- consumed by ConceptLoss, which aggregates *within* each document's own
        span, never merging documents that happen to share a window.
    known_labels: dense multi-hot ground-truth labels, broadcast to every token position of the
        document it belongs to -- this is what the bottleneck uses to compute the ground-truth
        known-concept embedding (for teacher forcing and as the reconstruction-loss target).
    """
    doc_spans = []
    known_labels = torch.zeros(
        len(starts), seq_len, n_concepts, device=device
    )  # shape: [B, T, n_concepts]
    for b, ws in enumerate(starts):
        for s, e, concept_ids in overlapping_docs(ws, ws + seq_len):
            doc_spans.append((b, s, e, concept_ids))
            known_labels[b, s:e, concept_ids] = (
                1.0  # broadcast this doc's labels over its own token span
            )
    return doc_spans, known_labels


def get_lr(step):
    """Linear warmup then cosine decay down to a floor."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def concept_contribution(head, k_hat, u_hat, epsilon, targets, mask):
    """Eq. 22: fraction of each masked target token's logit magnitude carried by the concept
    module (known + unknown) versus the residual. Eval-only diagnostic, not used in training --
    tracks whether the model is actually routing predictions through concepts."""
    k_logits, u_logits, eps_logits = head.decompose(
        k_hat, u_hat, epsilon
    )  # each: [B, T, vocab_size]
    # gather the logit of the actual target token at each position, then keep only masked positions
    k_term = (
        k_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[mask].abs()
    )  # [B, T, V] -> [B, T] -> [n_masked]
    u_term = u_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[mask].abs()
    eps_term = eps_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[mask].abs()
    return ((k_term + u_term) / (k_term + u_term + eps_term + 1e-8)).mean().item()


# ---------------------------------------------------------------------------
# Diffusion utilities (masking process, Sections 2.3, 5.2, 5.4.2)
# ---------------------------------------------------------------------------


def sample_noise_levels(num_blocks):
    """Per-block noise level t_b ~ U(0,1) (Section 5.4.2's per-block masking schedule). The
    paper's final model instead uses a moving Gaussian curriculum (center 0.2->0.8, sigma=0.3);
    swap this out for that later without touching corrupt()."""
    return torch.rand(num_blocks, device=device)


def corrupt(x0, block_len):
    """Independently mask tokens with a per-block noise level.

    x0: LongTensor [B, T], T must be divisible by block_len.
    Returns (x_t, mask): corrupted sequence and boolean mask of replaced positions.
    """
    B, T = x0.shape
    assert (
        T % block_len == 0
    ), "sequence length must be divisible by the diffusion block length"
    num_blocks = T // block_len

    t = sample_noise_levels(B * num_blocks).view(
        B, num_blocks
    )  # shape: [B*num_blocks] -> [B, num_blocks]
    t_per_token = t.repeat_interleave(
        block_len, dim=1
    )  # shape: [B, num_blocks] -> [B, T], broadcast each block's t to its tokens

    mask = (
        torch.rand(B, T, device=x0.device) < t_per_token
    )  # shape: [B, T], True where this token gets masked
    x_t = x0.clone()
    x_t[mask] = mask_token_id
    return x_t, mask


def build_block_causal_mask(total_len, block_len):
    """Bidirectional within a block, causal across blocks (Figure 16d)."""
    assert (
        total_len % block_len == 0
    ), "sequence length must be divisible by the diffusion block length"
    block_ids = (
        torch.arange(total_len, device=device) // block_len
    )  # shape: [total_len], each position's block index
    # mask[i, j] True means query i may attend to key j: allowed iff j's block <= i's block
    mask = block_ids.unsqueeze(1) >= block_ids.unsqueeze(
        0
    )  # shape: [total_len] -> [total_len, total_len]
    return mask


# ---------------------------------------------------------------------------
# Backbone (adapted from 1_gpt_tokens/gpt_light.py: same attention/feedforward/block structure,
# but attention takes an explicit block-causal attn_mask instead of a plain causal triangle)
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Grouped-query attention under an arbitrary attn_mask (here: block-causal)."""

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

    def forward(self, x, attn_mask):
        B, T, C = x.shape  # batch, sequence length, n_embed

        # project then split into per-head slices: [B, T, C] -> [B, T, heads, head_size] -> [B, heads, T, head_size]
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)

        # duplicate each kv head across the query heads that share it: [B, num_kv_heads, T, head_size] -> [B, num_heads, T, head_size]
        repeat_factor = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        # explicit attn_mask (not is_causal=True) since block-causal allows looking "ahead"
        # within the current block -- this disables the fused causal kernel on some backends,
        # so this backbone is slower per-step than a plain causal one at equal size
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # shape: [B, num_heads, T, head_size]

        out = (
            out.transpose(1, 2).contiguous().view(B, T, C)
        )  # merge heads back: -> [B, T, C]
        out = self.proj(out)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, n_embed, dropout):
        super().__init__()
        hidden_dim = int(
            4 * n_embed * 2 / 3
        )  # keep param count close to a standard 4x MLP despite the extra gate projection
        self.w1 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.silu(self.w1(x)) * self.w2(
            x
        )  # shape: [B, T, hidden_dim], elementwise gate
        out = self.w3(out)  # shape: [B, T, hidden_dim] -> [B, T, n_embed]
        out = self.dropout(out)
        return out


class Block(nn.Module):
    """One pre-norm transformer block: attention sub-layer, then feedforward sub-layer."""

    def __init__(self, n_embed, num_heads, dropout, num_kv_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa_head = MultiHeadAttention(
            num_heads, head_size, n_embed, dropout, num_kv_heads=num_kv_heads
        )
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, attn_mask):
        x = self.ln1(x)
        x = x + self.sa_head(x, attn_mask)  # residual connection around attention
        x = self.ln2(x)
        x = x + self.ffwd(x)  # residual connection around the feedforward
        return x


class TransformerBackbone(nn.Module):
    """Token/position embeddings through the block-causal transformer stack.

    Stops before any LM head: the head applies to the *bottlenecked* hidden state produced by
    ConceptBottleneck below, not this raw backbone output.
    """

    def __init__(
        self, vocab_size, n_embed, seq_len, num_heads, n_layers, dropout, num_kv_heads
    ):
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(seq_len, n_embed)
        self.blocks = nn.ModuleList(
            [Block(n_embed, num_heads, dropout, num_kv_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)

        self.apply(self._init_weights)
        # scale down residual-stream-writing projections so the residual stream doesn't blow up
        # in variance as depth increases (standard GPT-2-style init trick)
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

    def forward(self, idx, attn_mask):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)  # shape: [B, T] -> [B, T, n_embed]
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # shape: [T, n_embed]
        x = (
            token_emb + pos_emb
        )  # broadcast add: [B, T, n_embed] + [T, n_embed] -> [B, T, n_embed]
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.ln_f(x)
        return x  # shape: [B, T, n_embed]


# ---------------------------------------------------------------------------
# Concept bottleneck (Section 5.3, Eq. 5-8): h_bar = k_hat (known) + u_hat (unknown) + epsilon
# ---------------------------------------------------------------------------


def sparsify_top_k(activations, top_k):
    """Zero out every activation except the top-k per token (optional, off by default; matches
    the anneal-phase sparsification in the paper's Tables 25/26)."""
    if top_k is None or top_k >= activations.shape[-1]:
        return activations
    top_vals, top_idx = torch.topk(
        activations, top_k, dim=-1
    )  # shape: [..., n] -> [..., top_k]
    sparse = torch.zeros_like(activations)
    sparse.scatter_(
        -1, top_idx, top_vals
    )  # write the top-k values back into their original positions, rest stay 0
    return sparse


class SupervisedConceptHead(nn.Module):
    """Known concepts (size n, fixed by the concept library from ../3_atlas)."""

    def __init__(self, d, n, hidden_dim=None, p_cfg=0.1, top_k=None):
        super().__init__()
        hidden_dim = hidden_dim or d
        self.f = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n)
        )
        self.K = nn.Parameter(
            torch.randn(n, d) * 0.02
        )  # known concept embedding table, shape: [n, d]
        self.p_cfg = p_cfg
        self.top_k = top_k

    def forward(self, h):
        k = torch.sigmoid(
            self.f(h)
        )  # shape: [B, T, d] -> [B, T, n], per-token concept activations in [0, 1]
        k = sparsify_top_k(k, self.top_k)
        # classifier-free-guidance-style dropout on the known channel, so the model doesn't
        # become unusable if known concepts are later suppressed/absent at inference time
        k = nn.functional.dropout(k, p=self.p_cfg, training=self.training)
        k_hat = (
            k @ self.K
        )  # shape: [B, T, n] @ [n, d] -> [B, T, d], weighted sum of concept embeddings
        return k, k_hat

    def ground_truth_embedding(self, known_labels):
        # Eq. 11's k_hat_GT: weighted sum of K by the ground-truth chunk-level labels, broadcast
        # to every token position in the chunk (no dropout/top-k on the ground-truth path)
        return known_labels.float() @ self.K  # shape: [B, T, n] @ [n, d] -> [B, T, d]


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
            self.U = nn.Parameter(torch.randn(m, d) * 0.02)  # shape: [m, d]
        else:
            # U = A @ B factorization: cuts params from m*d to rank*(m+d) and turns the
            # per-token [m, d] matmul into two smaller ones -- worthwhile once m >> rank
            self.A = nn.Parameter(torch.randn(m, rank) * 0.02)  # shape: [m, rank]
            self.B = nn.Parameter(torch.randn(rank, d) * 0.02)  # shape: [rank, d]
        self.top_k = top_k

    def _embed(self, u):
        if self.rank is None:
            return u @ self.U  # shape: [B, T, m] @ [m, d] -> [B, T, d]
        return (
            u @ self.A
        ) @ self.B  # shape: [B, T, m] @ [m, rank] -> [B, T, rank] -> @ [rank, d] -> [B, T, d]

    def forward(self, h):
        u = torch.sigmoid(self.g(h))  # shape: [B, T, d] -> [B, T, m]
        u = sparsify_top_k(u, self.top_k)
        u_hat = self._embed(u)
        return u, u_hat


class ResidualModule(nn.Module):
    """epsilon = h - k_hat - u_hat, with dropout to discourage relying on the residual."""

    def __init__(self, p_epsilon=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p_epsilon)

    def forward(self, h, k_hat, u_hat):
        epsilon = h - k_hat - u_hat  # shape: [B, T, d], same shape as h
        return self.dropout(epsilon)


class ConceptBottleneck(nn.Module):
    """Composes the three heads and applies the Section 5.4.2 teacher-forcing substitution."""

    def __init__(
        self,
        d,
        n,
        unknown_ratio=3,
        m=None,
        p_cfg=0.1,
        p_epsilon=0.1,
        unknown_rank=None,
        top_k_known=None,
        top_k_unknown=None,
    ):
        super().__init__()
        self.n = n
        self.m = m or unknown_ratio * n
        self.known = SupervisedConceptHead(d, n, p_cfg=p_cfg, top_k=top_k_known)
        self.unknown = UnsupervisedConceptHead(
            d, self.m, rank=unknown_rank, top_k=top_k_unknown
        )
        self.residual = ResidualModule(p_epsilon)

    def forward(
        self,
        h,
        known_labels=None,
        teacher_force_known=False,
        teacher_force_unknown=False,
    ):
        k, k_hat = self.known(h)
        u, u_hat = self.unknown(h)

        k_hat_gt, u_hat_gt = None, None
        if known_labels is not None:
            k_hat_gt = self.known.ground_truth_embedding(
                known_labels
            )  # shape: [B, T, d]
            # u_hat_gt always derives from k_hat_gt (Eq. 11), never from predicted k_hat,
            # regardless of whether known-teacher-forcing is active this step
            u_hat_gt = h - k_hat_gt  # shape: [B, T, d]

        # teacher forcing: early in training, route the ground-truth concept embedding forward
        # instead of the (still unreliable) predicted one, so the LM head learns from a clean
        # signal before the concept heads have converged -- annealed down via teacher_forcing_alpha
        k_hat_used = (
            k_hat_gt if (teacher_force_known and k_hat_gt is not None) else k_hat
        )
        u_hat_used = (
            u_hat_gt if (teacher_force_unknown and u_hat_gt is not None) else u_hat
        )

        epsilon = self.residual(h, k_hat_used, u_hat_used)  # shape: [B, T, d]
        h_bar = k_hat_used + u_hat_used + epsilon  # shape: [B, T, d]

        intermediates = {
            "k": k,
            "u": u,
            "k_hat": k_hat,
            "u_hat": u_hat,  # predicted, pre-teacher-forcing (used by the losses)
            "k_hat_used": k_hat_used,
            "u_hat_used": u_hat_used,  # what actually formed h_bar
            "k_hat_gt": k_hat_gt,
            "u_hat_gt": u_hat_gt,
            "epsilon": epsilon,
        }
        return h_bar, intermediates


class ConceptLMHead(nn.Module):
    """Linear LM head on the bottlenecked hidden state (Eq. 8). No bias, so logits are an exact
    linear function of h_bar: decompose() below sums to exactly the same result as forward().
    """

    def __init__(self, d, vocab_size, tied_embedding=None):
        super().__init__()
        self.head = nn.Linear(d, vocab_size, bias=False)
        if tied_embedding is not None:
            self.head.weight = (
                tied_embedding  # share the tensor with the backbone's token embedding
            )

    def forward(self, h_bar):
        return self.head(h_bar)  # shape: [B, T, d] -> [B, T, vocab_size]

    def decompose(self, k_hat, u_hat, epsilon):
        # three matmuls against the same weight matrix; because the head has no bias, these
        # sum to exactly the same logits as forward(k_hat + u_hat + epsilon) (Eq. 8)
        return (
            self.head(k_hat),
            self.head(u_hat),
            self.head(epsilon),
        )  # each: [B, T, d] -> [B, T, vocab_size]


def teacher_forcing_alpha(
    step,
    max_steps,
    start=1.0,
    floor=0.5,
    warmup_frac=0.15,
    decay_end_frac=0.5,
    kind="cosine",
):
    """Holds at `start` through warmup, then decays (cosine or linear) to `floor` by
    decay_end_frac of training, then holds at `floor`. Defaults reproduce the paper's known-
    concept schedule (Table 33); pass warmup_frac=0.25, decay_end_frac=1.0, kind='linear' for
    the unknown-concept schedule.
    """
    warmup_steps_ = warmup_frac * max_steps
    decay_end_steps = decay_end_frac * max_steps
    if step <= warmup_steps_:
        return start
    if step >= decay_end_steps:
        return floor
    progress = (step - warmup_steps_) / (decay_end_steps - warmup_steps_)
    if kind == "cosine":
        return floor + 0.5 * (start - floor) * (1 + math.cos(math.pi * progress))
    if kind == "linear":
        return start - (start - floor) * progress
    raise ValueError(f"unknown schedule kind: {kind}")


class SteerlingModel(nn.Module):
    """Backbone + concept bottleneck + head, as one module (Section 5)."""

    def __init__(
        self,
        vocab_size,
        seq_len,
        n_embed,
        num_heads,
        num_kv_heads,
        n_layers,
        dropout,
        n_concepts,
        unknown_ratio=3,
        p_cfg=0.1,
        p_epsilon=0.1,
        unknown_rank=None,
        top_k_known=None,
        top_k_unknown=None,
        tie_weights=True,
    ):
        super().__init__()
        self.backbone = TransformerBackbone(
            vocab_size, n_embed, seq_len, num_heads, n_layers, dropout, num_kv_heads
        )
        self.bottleneck = ConceptBottleneck(
            n_embed,
            n_concepts,
            unknown_ratio=unknown_ratio,
            p_cfg=p_cfg,
            p_epsilon=p_epsilon,
            unknown_rank=unknown_rank,
            top_k_known=top_k_known,
            top_k_unknown=top_k_unknown,
        )
        # weight tying: sharing the token embedding with the LM head is a standard trick (roughly
        # halves the embedding-related parameter count) and, being one nn.Module tree, is
        # automatically deduplicated by model.parameters()/state_dict()
        tied_embedding = (
            self.backbone.token_embedding_table.weight if tie_weights else None
        )
        self.head = ConceptLMHead(n_embed, vocab_size, tied_embedding=tied_embedding)

    def forward(
        self,
        x_t,
        attn_mask,
        known_labels=None,
        teacher_force_known=False,
        teacher_force_unknown=False,
    ):
        h = self.backbone(x_t, attn_mask)  # shape: [B, T, n_embed]
        h_bar, intermediates = self.bottleneck(
            h,
            known_labels=known_labels,
            teacher_force_known=teacher_force_known,
            teacher_force_unknown=teacher_force_unknown,
        )
        logits = self.head(h_bar)  # shape: [B, T, vocab_size]
        return logits, intermediates


# ---------------------------------------------------------------------------
# Losses (Section 5.4.1, Eq. 2, 9-10, 12-15)
# ---------------------------------------------------------------------------


class LanguageModelingLoss(nn.Module):
    """Masked cross-entropy over masked positions only (Eq. 2), applied to h_bar's logits."""

    def forward(self, logits, targets, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        return F.cross_entropy(
            logits[mask], targets[mask]
        )  # shape: [n_masked, vocab] vs [n_masked]


class ConceptLoss(nn.Module):
    """OR-aggregated BCE over chunk-level (here: document-level) concept labels (Eq. 9-10).

    A document's label says "concept c appears somewhere in this document", not at every
    token, so we aggregate the per-token predicted activation into one per-document probability
    via a soft-OR before comparing to the binary label.
    """

    def forward(self, k, doc_spans):
        """
        k: [B, T, n] predicted known-concept activations.
        doc_spans: (batch_idx, tok_start, tok_end, concept_ids) per document overlapping the
            current window -- aggregation runs per document, over only its own token sub-range
            and against only its own labels, never merged across documents.
        """
        if not doc_spans:
            return torch.tensor(0.0, device=k.device)
        n = k.shape[-1]
        total = k.new_zeros(())
        for batch_idx, tok_start, tok_end, concept_ids in doc_spans:
            k_span = k[
                batch_idx, tok_start:tok_end, :
            ]  # shape: [B, T, n] -> [doc_len, n], this doc's tokens only
            k_chunk = 1 - torch.prod(
                1 - k_span, dim=0
            )  # shape: [doc_len, n] -> [n], Eq. 9 soft-OR aggregation
            y = k.new_zeros(
                n
            )  # shape: [n], multi-hot ground-truth label for this document
            y[concept_ids] = 1.0
            # clamp avoids log(0) in BCE when a prediction is fully saturated at 0 or 1
            total = total + F.binary_cross_entropy(
                k_chunk.clamp(1e-6, 1 - 1e-6), y, reduction="sum"
            )
        return total / len(doc_spans)


class ReconstructionLoss(nn.Module):
    """MSE between the unknown head's u_hat and its ground-truth target, masked positions only (Eq. 12)."""

    def forward(self, u_hat, u_hat_gt, mask):
        if u_hat_gt is None or mask.sum() == 0:
            return torch.tensor(0.0, device=u_hat.device)
        diff = u_hat[mask] - u_hat_gt[mask]  # shape: [B, T, d] -> [n_masked, d]
        return (diff**2).mean()


class IndependenceLoss(nn.Module):
    """Cross-covariance penalty between k_hat and u_hat (Eq. 13-14).

    Gradients flow only through the unknown side: the known-side features are detached, so the
    unknown head adapts to the (human-anchored) known head rather than the other way around.
    """

    def forward(self, k_hat, u_hat):
        d = k_hat.shape[-1]
        Hk = k_hat.detach().reshape(
            -1, d
        )  # shape: [B, T, d] -> [B*T, d], flatten batch+time into one "samples" axis
        Hu = u_hat.reshape(-1, d)  # shape: [B, T, d] -> [B*T, d]
        num_tokens = Hk.shape[0]

        Phi = Hk - Hk.mean(
            dim=0, keepdim=True
        )  # shape: [B*T, d], center each feature across the batch
        Psi = Hu - Hu.mean(dim=0, keepdim=True)  # shape: [B*T, d]
        cross_cov = (
            Psi.t() @ Phi
        )  # shape: [d, B*T] @ [B*T, d] -> [d, d], empirical cross-covariance
        return (cross_cov**2).sum() / (
            d**2 * max(num_tokens - 1, 1)
        )  # normalized Frobenius norm^2


class SteerlingLoss(nn.Module):
    """Composite: L = L_LM + lambda_concept*L_concept + lambda_rec*L_rec + lambda_indep*L_indep (Eq. 15)."""

    def __init__(self, lambda_concept=1.0, lambda_rec=1.0, lambda_indep=1.0):
        super().__init__()
        self.lm_loss = LanguageModelingLoss()
        self.concept_loss = ConceptLoss()
        self.rec_loss = ReconstructionLoss()
        self.indep_loss = IndependenceLoss()
        self.lambda_concept = lambda_concept
        self.lambda_rec = lambda_rec
        self.lambda_indep = lambda_indep

    def forward(self, logits, targets, mask, intermediates, doc_spans):
        l_lm = self.lm_loss(logits, targets, mask)
        l_concept = self.concept_loss(intermediates["k"], doc_spans)
        l_rec = self.rec_loss(intermediates["u_hat"], intermediates["u_hat_gt"], mask)
        l_indep = self.indep_loss(intermediates["k_hat"], intermediates["u_hat"])

        total = (
            l_lm
            + self.lambda_concept * l_concept
            + self.lambda_rec * l_rec
            + self.lambda_indep * l_indep
        )

        components = {
            "lm": l_lm.item(),
            "concept": l_concept.item(),
            "rec": l_rec.item(),
            "indep": l_indep.item(),
        }
        return total, components


# ---------------------------------------------------------------------------
# Instantiate + train
# ---------------------------------------------------------------------------

attn_mask = build_block_causal_mask(
    seq_len, diff_block_len
)  # shape: [seq_len, seq_len], shared across all steps
model = SteerlingModel(
    vocab_size,
    seq_len,
    n_embed,
    num_heads,
    num_kv_heads,
    n_layers,
    dropout,
    n_concepts,
    unknown_ratio=unknown_ratio,
    p_cfg=p_cfg,
    p_epsilon=p_epsilon,
    unknown_rank=unknown_rank,
).to(device)
loss_fn = SteerlingLoss(lambda_concept, lambda_rec, lambda_indep)

print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")


@torch.no_grad()
def estimate_loss():
    """Average loss (+ components + concept contribution) over eval_iters fresh val batches,
    so the number printed at each eval step is a smoothed estimate rather than one noisy batch.
    """
    model.eval()
    totals = {
        "loss": 0.0,
        "lm": 0.0,
        "concept": 0.0,
        "rec": 0.0,
        "indep": 0.0,
        "contribution": 0.0,
    }
    for _ in range(eval_iters):
        x0, starts = get_batch("val")
        x_t, mask = corrupt(x0, diff_block_len)
        doc_spans, known_labels = build_supervision(starts)

        logits, intermediates = model(x_t, attn_mask, known_labels=known_labels)
        total_loss, components = loss_fn(logits, x0, mask, intermediates, doc_spans)

        totals["loss"] += total_loss.item()
        for key in ("lm", "concept", "rec", "indep"):
            totals[key] += components[key]
        totals["contribution"] += concept_contribution(
            model.head,
            intermediates["k_hat_used"],
            intermediates["u_hat_used"],
            intermediates["epsilon"],
            x0,
            mask,
        )
    model.train()
    return {k: v / eval_iters for k, v in totals.items()}


if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded existing checkpoint from {ckpt_path}, skipping training.")
else:
    print("No existing checkpoint found, training from scratch.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for step in range(max_steps + 1):
        x0, starts = get_batch("train")
        x_t, mask = corrupt(x0, diff_block_len)
        doc_spans, known_labels = build_supervision(starts)

        # anneal teacher forcing from full (step 0) down to each schedule's floor -- see
        # teacher_forcing_alpha's docstring for the known/unknown schedule shapes
        alpha_known = teacher_forcing_alpha(
            step, max_steps, warmup_frac=0.15, decay_end_frac=0.5, kind="cosine"
        )
        alpha_unknown = teacher_forcing_alpha(
            step, max_steps, warmup_frac=0.25, decay_end_frac=1.0, kind="linear"
        )
        teacher_force_known = torch.rand(()).item() < alpha_known
        teacher_force_unknown = torch.rand(()).item() < alpha_unknown

        logits, intermediates = model(
            x_t,
            attn_mask,
            known_labels=known_labels,
            teacher_force_known=teacher_force_known,
            teacher_force_unknown=teacher_force_unknown,
        )
        total_loss, components = loss_fn(logits, x0, mask, intermediates, doc_spans)

        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % eval_interval == 0:
            val = estimate_loss()
            print(
                f"step {step}, train loss {total_loss.item():.4f} "
                f"(lm {components['lm']:.4f} concept {components['concept']:.4f} "
                f"rec {components['rec']:.4f} indep {components['indep']:.4f}) | "
                f"val loss {val['loss']:.4f}, concept contribution {val['contribution']:.3f}, "
                f"lr {current_lr:.6f}"
            )

    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Training complete, saved checkpoint to {ckpt_path}")


@torch.no_grad()
def generate():
    """Basic random-remasking MDM sampler for sanity-checking output only -- not the paper's
    efficient block-wise KV-cached inference procedure (out of scope for this script).
    """
    model.eval()
    x = torch.full(
        (1, seq_len), mask_token_id, dtype=torch.long, device=device
    )  # shape: [1, seq_len], start fully masked
    masked = torch.ones_like(
        x, dtype=torch.bool
    )  # shape: [1, seq_len], tracks which positions are still masked

    for step in range(1, gen_steps + 1):
        logits, _ = model(x, attn_mask)  # shape: [1, seq_len, vocab_size]
        target_masked_count = round(
            seq_len * (1 - step / gen_steps)
        )  # shrink the masked budget linearly over gen_steps

        probs = F.softmax(
            logits / gen_temperature, dim=-1
        )  # shape: [1, seq_len, vocab_size]
        if gen_top_k is not None:
            v, _ = torch.topk(
                logits, min(gen_top_k, logits.size(-1)), dim=-1
            )  # shape: [1, seq_len, top_k]
            probs = torch.where(
                logits < v[..., [-1]], torch.zeros_like(probs), probs
            )  # zero out everything below the k-th largest logit
            probs = probs / probs.sum(
                dim=-1, keepdim=True
            )  # renormalize after truncation
        sampled = torch.multinomial(probs.view(-1, vocab_size), 1).view(
            1, seq_len
        )  # shape: [seq_len, vocab_size] -> [seq_len, 1] -> [1, seq_len]

        masked_positions = masked[0].nonzero(as_tuple=True)[
            0
        ]  # shape: [n_still_masked], indices of masked positions
        num_to_reveal = max(len(masked_positions) - target_masked_count, 0)
        if num_to_reveal > 0:
            # reveal a random subset of currently-masked positions (not necessarily the most
            # confident ones) -- simplest possible sampler, good enough for a sanity check
            reveal_idx = masked_positions[
                torch.randperm(len(masked_positions))[:num_to_reveal]
            ]
            x[0, reveal_idx] = sampled[0, reveal_idx]
            masked[0, reveal_idx] = False

    model.train()
    return x[0].tolist()


print(decode(generate()))
