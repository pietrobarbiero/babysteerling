import bisect
import json
import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from torch_concepts.nn import BaseConceptLayer
from babysteerling.data.utils import load_lifted_token_prototypes

torch.manual_seed(1337)

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

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

# Concept bottleneck params
top_k_known = 25      # set an int to sparsify known-concept activations per token
top_k = 5    # set an int to sparsify unknown-concept activations per token
lambda_concept = 0.0  # lambda_concept: weight of the concept loss relative to LM loss

ckpt_dir = "./model"
ckpt_path = os.path.join(ckpt_dir, "gpt_proto_steerling.pt")

# data: reuse 3_atlas's concept-annotated TinyStories (tokens + per-document concept labels)
data_dir = "../experiments/data"
tokens_path = os.path.join(data_dir, "steerling_tokens.pt")
concepts_path = os.path.join(data_dir, "steerling_concepts.pt")
concept_lib_path = os.path.join(data_dir, "concepts.json")
tokenizer_path = os.path.join(data_dir, "tokenizer.json")
lifted_tokens_positive_path = os.path.join(data_dir, "lifted_tokens.json")
lifted_tokens_negative_path = os.path.join(data_dir, "lifted_tokens_negative.json")

tok = Tokenizer.from_file(tokenizer_path)
vocab_size = tok.get_vocab_size()
decode = lambda ids: tok.decode(ids)

data = torch.load(tokens_path)
doc_records = torch.load(concepts_path)  # [{chunk_id, start, end, concept_ids}, ...]
with open(concept_lib_path) as f:
    concept_library = json.load(f)
n_concepts = len(concept_library)
print(f"Loaded {len(data)} tokens, {len(doc_records)} documents, {n_concepts} known concepts")
concept_ids = [c for c in range(n_concepts)]
proto_token_ids = load_lifted_token_prototypes(data_dir, concept_ids)

n = int(0.9 * len(data))
# documents are laid out contiguously and non-overlapping, so a sorted list of starts lets us
# binary-search the small contiguous range overlapping any sampled window
doc_starts = [d['start'] for d in doc_records]


def overlapping_docs(window_start, window_end):
    i = max(bisect.bisect_right(doc_starts, window_start) - 1, 0)
    spans = []
    while i < len(doc_records) and doc_records[i]['start'] < window_end:
        d = doc_records[i]
        s, e = max(d['start'], window_start), min(d['end'], window_end)
        if e > s:
            spans.append((s - window_start, e - window_start, d['concept_ids']))
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
    lo, hi = (0, n) if split == 'train' else (n, len(data))
    ix = torch.randint(lo, hi - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
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
    backbone.eval(); bottleneck.eval(); head.eval()
    for split in ('train', 'val'):
        totals = {'total': 0.0, 'lm': 0.0, 'concept': 0.0}
        for _ in range(eval_iters):
            xb, yb, starts = get_batch(split)
            loss, components = compute_loss(xb, yb, starts)
            totals['total'] += loss.item()
            for key in ('lm', 'concept'):
                totals[key] += components[key]
        out[split] = {key: value / eval_iters for key, value in totals.items()}
    backbone.train(); bottleneck.train(); head.train()
    return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, n_embed, block_size, dropout, num_kv_heads=None):
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
            q, k, v,
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
        self.sa_head = MultiHeadAttention(num_heads, head_size, n_embed, block_size, dropout, num_kv_heads=num_kv_heads)
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
    any LM head -- the head applies to the bottlenecked hidden state, not this raw output."""

    def __init__(self, vocab_size, n_embed, block_size, num_heads, n_layers, dropout, num_kv_heads):
        super().__init__()
        self.block_size = block_size
        self.n_layers = n_layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, block_size, num_heads, dropout, num_kv_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('w3.weight'):
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


class LinearSelector(nn.Module):
    """Selector: score every concept with one Linear(d, Kt), take the candidates_per_token
    highest-scoring ones. Cost O(d*Kt) per token, same as the dense baseline. Use this while Kt
    is small enough for a dense [d, Kt] projection to be cheap; switch to ProductKeySelector
    once Kt grows past that.

    forward(x) -> (idx, weight): idx is [B, T, candidates_per_token] concept ids. weight is the
    same shape and is exactly 1.0 in the forward pass (see module docstring), carrying gradient
    back into `score`.

    Training-time noise (same idea as ProductKeySelector): top-k's backward gives 0 gradient to
    every concept not selected, so without noise a concept that isn't in the top-k at
    initialization can never be discovered, since nothing ever pushes its score up. Adding noise
    before top-k lets not-yet-selected concepts get sampled in occasionally, so `score` can
    actually learn about them.
    """

    def __init__(self, in_embeddings, out_concepts, candidates_per_token=25):
        super().__init__()
        self.candidates_per_token = candidates_per_token
        self.score = nn.Linear(in_embeddings, out_concepts)
        self.noise = nn.Linear(in_embeddings, out_concepts)

    def forward(self, x):
        scores = self.score(x)  # shape: [B, T, Kt]
        if self.training:
            scores = scores + torch.randn_like(scores) * F.softplus(self.noise(x))
        vals, idx = scores.topk(self.candidates_per_token, dim=-1)  # shape: [B, T, C] (both)
        # forward value is exactly 1.0 (detach blocks gradient, not the arithmetic); backward
        # flows through sigmoid(v)
        weight = (torch.ones_like(vals) - torch.sigmoid(vals)).detach() + torch.sigmoid(vals)
        return idx, weight


class _RunningCenter(nn.Module):
    """Centers its input by subtracting a running per-feature mean, tracked via a fixed EMA rule
    over training -- not a learned parameter, so it doesn't need gradient from any loss to find
    the correction (LiftedTokenPredictor gets none when model.use_concept_loss=False). Instead it
    measures the offset directly from the forward activations, which is what actually fixes
    LiftedTokenPredictor's cosine-similarity collapse: trained transformers commonly develop a
    handful of "massive activation" dimensions with large, roughly token-independent magnitude,
    which dominate a raw dot product regardless of genuine token content. A learned bias could in
    principle cancel the same offset, but only if backprop happens to push it there; this doesn't
    wait on that, and works from a freshly-initialized backbone just as well as a trained one --
    it just tracks whatever offset currently exists, live, throughout training.

    Deliberately not BatchNorm: forward() always subtracts the slower-moving running_mean, never
    the current batch's own statistics (which only nudge running_mean, see update()), so it's
    insensitive to batch size/composition and behaves identically in train() and eval() -- no
    train/eval discrepancy, no small/uneven-batch instability. No variance normalization either:
    a near-constant offset (large mean, low variance -- the "massive activation" signature) is
    fully corrected by centering alone, so there's nothing extra to gain from also rescaling.
    """

    def __init__(self, dim, momentum=0.1):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(dim))

    def update(self, x):
        """Nudges running_mean toward this batch's mean. Call only on the larger, more
        representative stream (LiftedTokenPredictor's per-token query x, not its much smaller
        per-candidate prototype batch) so the estimate isn't skewed by a small sample."""
        if self.training:
            with torch.no_grad():
                batch_mean = x.detach().reshape(-1, x.shape[-1]).mean(dim=0)
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean, alpha=self.momentum)

    def forward(self, x):
        return x - self.running_mean


class LiftedTokenPredictor(nn.Module):
    """Predictor: like PrototypePredictor, but scores candidates against each concept's own top-k
    positive/negative lifted tokens (babysteerling.data.babyatlas.compute_lifted_tokens) instead
    of LLM-generated prototype sentences (concept_prototypes.json). Each prototype is a single
    real corpus token, not a multi-token sentence, so there's no sequence to average-pool: a
    prototype's backbone encoding (squeezed, not _masked_mean_pool'd) is used directly. No
    "unrelated" category either (unlike PrototypePredictor's 3-way split) -- a lifted token is by
    construction one of exactly two things, so the value axis is just [-1, +1].

    Aggregation is deliberately not a softmax-weighted average over every prototype the way
    PrototypePredictor's is. Within the positive group and the negative group independently, only
    the single best-matching prototype counts: forward value is the true max of that group's
    softmax weights; backward flows through the sum-of-squares of those weights instead (a
    straight-through estimator -- plain torch.max on a softmax output isn't literally
    zero-gradient elsewhere, softmax's own coupling already smears some gradient to every
    candidate, but it still vanishes with a candidate's own weight; sum-of-squares doesn't remove
    that either, it's just the simplest function of the weights that still reaches every
    candidate rather than only the argmax's own path). The two groups' maxes are then combined
    directly: score = (max_pos - max_neg) / (max_pos + max_neg), a convex combination of exactly
    +1 and -1 by the (renormalized) positive/negative maxes -- not routed through PrototypePredictor's
    self.value @ wei_p, since there's no third ("unrelated") category to combine over here.

    forward(x, idx, weight) -> k_dense: idx/weight are both [B, T, C].
    """

    _NEG_MASK_VALUE = -1e9  # additive mask: large-but-finite, so an all-padded group softmaxes to
    # uniform (not NaN) -- true -inf can leak NaN into the backward pass even through a later
    # masked_fill/where, since softmax's own backward formula uses its (possibly all-NaN) output

    def __init__(self, in_embeddings, out_concepts, proto_token_ids, backbone, top_k=5):
        super().__init__()
        d = in_embeddings
        Kt = out_concepts
        assert proto_token_ids.shape[0] == Kt, (
            f"proto_token_ids has {proto_token_ids.shape[0]} concepts, expected {Kt}"
        )
        assert tuple(proto_token_ids.shape[1:]) == (2, top_k), (
            f"proto_token_ids shape {tuple(proto_token_ids.shape[1:])} != (2, {top_k}) "
            "(axis 1: 0=negative, 1=positive -- see data.utils.load_lifted_token_prototypes)"
        )
        self.Kt = Kt
        self.top_k = top_k

        self.backbone = backbone  # shared with the main model; see module docstring
        self.register_buffer('proto_token_ids', proto_token_ids, persistent=False)

        self.register_buffer('value', torch.tensor([-1.] * top_k + [1.] * top_k), persistent=False)
        # bias gives proto_query some capacity to cancel a shared, roughly token-independent
        # offset in x/hidden on its own; _RunningCenter (see class above forward()) is the
        # primary fix, tracking that offset directly instead of waiting for gradient to find it
        self.proto_query = nn.Linear(d, d, bias=True)
        self.center = _RunningCenter(d)
        # cosine similarity between unit vectors concentrates tightly around 0 in high dimensions
        # (std ~= 1/sqrt(d)), so softmax over raw cosine sims is nearly uniform regardless of how
        # well-matched a candidate actually is. A learned temperature (CLIP's logit_scale trick)
        # lets softmax sharpen as needed while staying bounded -- clamped in forward() so it can
        # never grow back into the unbounded-magnitude problem cosine similarity was meant to fix.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, x, idx, weight):
        B, T = x.shape[:2]

        # encode every concept's prototypes, not just the ones some token in this batch actually
        # picked: Kt is fixed and small (see nn.bottleneck), while the "only encode what's
        # selected" version made the backbone's input batch shaped by U = number of distinct
        # selected concepts, which changes every step. MPS's allocator crashes with SIGTRAP when
        # a buffer's shape keeps churning across iterations like that (pytorch/pytorch#178079);
        # a fixed [Kt, ...] shape every call sidesteps it entirely.
        #
        # validity from the raw (possibly -1 = "no lifted token here") ids, before any clamping --
        # token id 0 is '<|endoftext|>', a real token that can legitimately be a lifted token
        # itself, so it must not be mistaken for padding (see data.utils.load_lifted_token_prototypes)
        valid_all = self.proto_token_ids >= 0  # shape: [Kt, 2, top_k]
        proto_ids_safe = self.proto_token_ids.clamp(min=0)  # -1 -> 0, a valid (if dummy) embedding id;
        # only used for the lookup below -- its actual embedding never matters, since invalid
        # slots are masked out of the softmax before they can affect anything

        # each prototype is exactly one token: a length-1 sequence per prototype, so there's no
        # sentence to average-pool -- squeeze the trivial length-1 axis back out instead
        hidden = self.backbone(proto_ids_safe.reshape(-1, 1))  # shape: [Kt*2*top_k, 1, d]
        hidden = hidden.squeeze(1).view(self.Kt, 2, self.top_k, -1)  # [Kt, 2, top_k, d]
        # x and h are compared in the exact same learned space (same proto_query for both), so
        # nothing bounds the raw dot product's magnitude -- gradients on proto_query get pulled
        # from both its "query" and "key" role every step, reinforcing growth along its own
        # dominant direction. Cosine similarity removes that degree of freedom: only the angle
        # between x and h matters, not how large proto_query's weights have grown.
        # self.center strips a shared offset before normalizing (see _RunningCenter) -- applied
        # to both x and hidden identically, since they're compared in the same space
        key_all = F.normalize(self.center(self.proto_query(hidden)), dim=-1)  # shape: [Kt, 2, top_k, H]

        key_cand = key_all[idx]  # shape: [B, T, C, 2, top_k, H]
        valid_cand = valid_all[idx]  # shape: [B, T, C, 2, top_k]

        q_raw = self.proto_query(x)  # shape: [B, T, H]
        self.center.update(q_raw)  # x is the larger, more representative stream -- see update()
        q = F.normalize(self.center(q_raw), dim=-1)  # shape: [B, T, H]
        raw = torch.einsum('bth,btcgkh->btcgk', q, key_cand)  # [B, T, C, 2, top_k], cosine similarity in [-1, 1]
        # clamp the log-value, not exp()'s result: exp() overflows to inf before a post-hoc clamp
        # could catch it, and inf's gradient combined with clamp's zero-grad region there
        # produces 0*inf = nan for logit_scale -- silently reintroducing the same failure this
        # temperature exists to prevent.
        scale = self.logit_scale.clamp(max=math.log(100)).exp()  # bounded temperature, see __init__
        raw = raw * scale
        raw = raw.masked_fill(~valid_cand, self._NEG_MASK_VALUE)
        wei_p = raw.flatten(-2, -1).softmax(dim=-1)  # [B, T, C, 2*top_k]

        # convex combination of exactly +1 (positive) and -1 (negative), weighted by the
        # renormalized (max_pos, max_neg) pair; eps guards the (should-be-rare) case where a
        # concept has no valid tokens in either direction at all
        candidate_score = (wei_p @ self.value) * weight  # shape: [B, T, C, 2*top_k] @ [2*top_k] -> [B, T, C]

        k_dense = x.new_zeros(B, T, self.Kt)
        k_dense.scatter_add_(dim=-1, index=idx, src=candidate_score)  # 0 everywhere not selected
        return k_dense


class ConceptBottleneck(BaseConceptLayer):
    """Combines any Selector (LinearSelector, ProductKeySelector) with a PrototypePredictor
    into the activation()/embed()/forward()/ground_truth_embedding()/.K interface
    ConceptBottleneck expects. Swap `selector` without touching how candidates get scored.
    """

    def __init__(self, in_embeddings, out_concepts, selector, predictor):
        super().__init__(out_concepts=out_concepts, in_concepts=None, in_embeddings=in_embeddings)
        self.selector = selector
        self.predictor = predictor
        self.K = nn.Parameter(
            torch.randn(self.out_concepts_shape, self.in_embeddings_shape) * 0.02
        )  # value/reconstruction table, as in SparseEmbeddingToConcept

    def forward(self, x):
        idx, weight = self.selector(x)
        return self.predictor(x, idx, weight)



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
            total = total + F.binary_cross_entropy(k_chunk.clamp(1e-6, 1 - 1e-6), y, reduction='sum')
        return total / len(doc_spans)


class ConceptLMHead(nn.Module):
    """No bias, so logits are an exact linear function of h_bar: decompose() sums to forward()."""

    def __init__(self, n, vocab_size, tied_embedding=None):
        super().__init__()
        self.head = nn.Linear(n, vocab_size, bias=False)

    def forward(self, h_bar):
        return self.head(h_bar)

    def decompose(self, k_hat, u_hat, epsilon):
        return self.head(k_hat)


backbone = TransformerModel(vocab_size, n_embed, block_size, num_heads, n_layers, dropout, num_kv_heads).to(device)
selector = LinearSelector(n_embed, n_concepts, candidates_per_token=top_k_known).to(device)
predictor = LiftedTokenPredictor(
    n_embed, n_concepts,
    proto_token_ids=proto_token_ids,
    backbone=backbone,
    top_k=top_k,
)
bottleneck = ConceptBottleneck(
    n_embed, n_concepts,
    selector=selector,
    predictor=predictor,
).to(device)
head = ConceptLMHead(n_concepts, vocab_size, tied_embedding=backbone.token_embedding_table.weight).to(device)

concept_loss_fn = ConceptLoss()

# dedupe: head.weight is tied to backbone's token embedding, so naively concatenating each
# module's .parameters() would list that tensor twice and double-step it in the optimizer
all_params = list(dict.fromkeys(
    list(backbone.parameters()) + list(bottleneck.parameters()) + list(head.parameters())
))
print(sum(p.numel() for p in all_params) / 1e6, 'M params')


def compute_loss(xb, yb, starts):
    doc_spans, known_labels = build_supervision(starts)
    h = backbone(xb)
    h_bar = bottleneck(h)
    logits = head(h_bar)

    B, T, C = logits.shape
    lm_loss = F.cross_entropy(logits.view(B * T, C), yb.view(B * T))
    concept_loss = concept_loss_fn(h_bar, doc_spans)

    total_loss = lm_loss + lambda_concept * concept_loss
    components = {
        'lm': lm_loss.item(),
        'concept': concept_loss.item(),
    }
    return total_loss, components


if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt['backbone'])
    bottleneck.load_state_dict(ckpt['bottleneck'])
    head.load_state_dict(ckpt['head'])
    print(f"Loaded existing checkpoint from {ckpt_path}, skipping training.")
else:
    print("No existing checkpoint found, training from scratch.")

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    for step in range(max_steps + 1):
        xb, yb, starts = get_batch('train')

        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        loss, _ = compute_loss(xb, yb, starts)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()

        if step % eval_interval == 0:
            losses = estimate_loss()

            def fmt(split):
                s = losses[split]
                return (f"{s['total']:.4f} (lm {s['lm']:.4f} concept {s['concept']:.4f})")

            print(f"step {step}, train loss: {fmt('train')}, val loss: {fmt('val')}, lr: {current_lr:.6f}")

    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        'backbone': backbone.state_dict(),
        'bottleneck': bottleneck.state_dict(),
        'head': head.state_dict(),
    }, ckpt_path)
    print(f"Training complete, saved checkpoint to {ckpt_path}")


@torch.no_grad()
def generate(idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        h = backbone(idx_cond)
        h_bar = bottleneck(h)
        logits = head(h_bar)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# generate a sample
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)[0].tolist()))
