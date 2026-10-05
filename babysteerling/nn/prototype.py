import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from torch_concepts.nn import BaseConceptLayer

from .encoder import sparsify_top_k


def _masked_mean_pool(hidden, token_ids):
    """Averages a per-token representation into one vector per prototype, ignoring pad (id 0)
    positions. hidden: [..., Tp, d], already computed (a table lookup for
    PrototypeCrossAttention, a full backbone pass for PrototypePredictor). token_ids: [..., Tp],
    used only to build the pad mask.
    """
    mask = (token_ids != 0).unsqueeze(-1)  # shape: [..., Tp, 1]
    summed = (hidden * mask).sum(dim=-2)  # shape: [..., d]
    counts = mask.sum(dim=-2).clamp(min=1)  # shape: [..., 1]
    return summed / counts


def _fixed_value_axis(proto_token_ids):
    """proto_token_ids: [Kt, P, Tp], P = 3 * per_type (negative, unrelated, positive, in that
    order; see data.utils.load_concept_prototype_tokens). Returns the fixed value buffer
    [-1]*per_type + [0]*per_type + [1]*per_type."""
    per_type = proto_token_ids.shape[1] // 3
    return torch.tensor([-1.0] * per_type + [0.0] * per_type + [1.0] * per_type)


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
        vals, idx = scores.topk(
            self.candidates_per_token, dim=-1
        )  # shape: [B, T, C] (both)
        # forward value is exactly 1.0 (detach blocks gradient, not the arithmetic); backward
        # flows through sigmoid(v)
        weight = (torch.ones_like(vals) - torch.sigmoid(vals)).detach() + torch.sigmoid(
            vals
        )
        return idx, weight


class ProductKeySelector(nn.Module):
    """Selector: factorized row/col retrieval. Top row keys x top col keys give
    Ktr*Ktr = topk_axis^2 candidate concept ids, with queries derived from the token's own
    hidden state. Never touches all Kt concepts: cost is O(sqrt(Kt)) for the two axis lookups
    plus O(topk_axis^2) for the candidate pool.

    Which concept lands at which (row, col) is fixed and sequential (concept_id = row*Ktr +
    col). The row/col keys themselves are learned, but a token's query can only ever reach
    candidates in that fixed grid.

    forward(x) -> (idx, weight): idx is [B, T, topk_axis**2] concept ids. weight is the same
    shape and is exactly 1.0 in the forward pass (see module docstring). Without it,
    row_keys/col_keys/row_query/col_query would never get any gradient (top-k's indices carry
    none) and would stay at their random init forever.
    """

    def __init__(self, in_embeddings, out_concepts, topk_axis=5, key_dim=None):
        super().__init__()
        d = in_embeddings
        Kt = out_concepts
        self.Kt = Kt

        self.Ktr = math.isqrt(Kt)
        while self.Ktr * self.Ktr < Kt:
            self.Ktr += 1  # grid must cover every concept id even when Kt isn't a perfect square
        assert (
            topk_axis < self.Ktr
        ), f"topk_axis ({topk_axis}) must be smaller than Ktr ({self.Ktr})"
        self.topk_axis = (
            topk_axis  # this alone determines sparsity (topk_axis**2 candidates)
        )
        H = key_dim or min(d, 32)  # a routing key doesn't need the full hidden width

        self.row_keys = nn.Parameter(torch.randn(self.Ktr, H) * 0.02)
        self.col_keys = nn.Parameter(torch.randn(self.Ktr, H) * 0.02)
        self.row_query = nn.Linear(d, H, bias=False)
        self.col_query = nn.Linear(d, H, bias=False)
        self.row_noise = nn.Linear(d, self.Ktr)
        self.col_noise = nn.Linear(d, self.Ktr)

    def forward(self, x):
        B, T = x.shape[:2]
        row_scores = self.row_query(x) @ self.row_keys.T  # shape: [B, T, Ktr]
        col_scores = self.col_query(x) @ self.col_keys.T

        if self.training:
            # noisy top-k for exploration (Shazeer et al. 2017's noisy gating); off at eval so
            # inference is deterministic given fixed weights
            row_scores = row_scores + torch.randn_like(row_scores) * F.softplus(
                self.row_noise(x)
            )
            col_scores = col_scores + torch.randn_like(col_scores) * F.softplus(
                self.col_noise(x)
            )

        row_vals, row_idx = row_scores.topk(
            self.topk_axis, dim=-1
        )  # shape: [B, T, topk_axis] (both)
        col_vals, col_idx = col_scores.topk(self.topk_axis, dim=-1)

        idx = row_idx.unsqueeze(-1) * self.Ktr + col_idx.unsqueeze(
            -2
        )  # shape: [B, T, topk_axis, topk_axis]
        idx = idx.reshape(B, T, -1)  # shape: [B, T, topk_axis**2]

        row_ste = (
            torch.ones_like(row_vals) - torch.sigmoid(row_vals)
        ).detach() + torch.sigmoid(row_vals)
        col_ste = (
            torch.ones_like(col_vals) - torch.sigmoid(col_vals)
        ).detach() + torch.sigmoid(col_vals)
        weight = (row_ste.unsqueeze(-1) * col_ste.unsqueeze(-2)).reshape(
            B, T, -1
        )  # exactly 1.0 forward

        # Ktr**2 can exceed Kt when Kt isn't a perfect square; clamp out-of-range grid corners
        # onto the last concept id instead of indexing out of bounds
        return idx.clamp(max=self.Kt - 1), weight


class PrototypePredictor(nn.Module):
    """Predictor: scores candidate concept ids (from any Selector) against their own prototypes
    and writes the result into a dense [B, T, Kt] activation. Works the same whether idx came
    from LinearSelector or ProductKeySelector.

    Only the concepts some token actually selected get encoded, deduped with torch.unique first
    (encoding runs the full backbone, not a cheap table lookup). PrototypeCrossAttention uses a
    table lookup instead, since it touches all Kt concepts every call.

    On MPS, the encoded batch is padded to a power-of-two bucket (see forward()) to avoid a
    shape-churn crash, without changing torch.unique's actual dedup.

    forward(x, idx, weight) -> k_dense: idx/weight are both [B, T, C].
    """

    def __init__(
        self, in_embeddings, out_concepts, proto_token_ids, backbone, key_dim=None
    ):
        super().__init__()
        d = in_embeddings
        Kt = out_concepts
        assert (
            proto_token_ids.shape[0] == Kt
        ), f"proto_token_ids has {proto_token_ids.shape[0]} concepts, expected {Kt}"
        self.Kt = Kt
        H = key_dim or min(d, 32)  # a scoring key doesn't need the full hidden width

        self.backbone = backbone  # shared with the main model; see module docstring
        proto_token_ids = proto_token_ids.reshape(
            Kt, -1, proto_token_ids.shape[-1]
        )  # [Kt, P, Tp]
        Tp = proto_token_ids.shape[-1]
        assert Tp <= backbone.block_size, (
            f"prototype tokens ({Tp}) exceed the backbone's block_size ({backbone.block_size}); "
            "reduce max_prototype_tokens or increase block_size"
        )
        self.register_buffer("proto_token_ids", proto_token_ids, persistent=False)
        self.register_buffer(
            "value", _fixed_value_axis(proto_token_ids), persistent=False
        )

        self.proto_query = nn.Linear(d, H, bias=False)
        self.proto_key = nn.Linear(d, H, bias=False)

    def forward(self, x, idx, weight):
        B, T = x.shape[:2]
        C = idx.shape[-1]
        P, Tp = self.proto_token_ids.shape[1:]

        # only encode the distinct concepts some token actually picked, each at most once
        unique_idx, inverse = torch.unique(
            idx, return_inverse=True
        )  # unique_idx: [U]; inverse: [B, T, C]
        proto_ids_selected = self.proto_token_ids[unique_idx]  # shape: [U, P, Tp]

        # always causal: prototypes are static reference text being read, not generated
        hidden = self.backbone(
            proto_ids_selected.reshape(-1, Tp)
        )  # shape: [U*P, Tp, d]
        hidden = hidden.reshape(unique_idx.shape[0], P, Tp, -1)  # shape: [U, P, Tp, d]
        proto_emb_selected = _masked_mean_pool(
            hidden, proto_ids_selected
        )  # shape: [U, P, d]
        key_selected = self.proto_key(proto_emb_selected)  # shape: [U, P, H]

        if x.device.type == "mps":
            # MPS crashes (SIGTRAP, in its buffer allocator) when this gather's backward has to
            # scatter into a buffer shaped by U, which changes every step. Route through a fixed
            # [Kt, P, H] buffer instead -- only cheap because Kt is small here (linear_selector).
            key_all = key_selected.new_zeros(self.Kt, P, key_selected.shape[-1])
            key_all[unique_idx] = key_selected
            key_cand = key_all[idx]  # shape: [B, T, C, P, H]
        else:
            key_cand = key_selected[inverse]  # shape: [B, T, C, P, H]

        q = self.proto_query(x)  # shape: [B, T, H]
        wei = torch.einsum("bth,btcph->btcp", q, key_cand) / (
            key_cand.shape[-1] ** 0.5
        )  # shape: [B, T, C, P]
        wei_p = wei.softmax(
            dim=-1
        )  # normalize over this candidate's own P prototypes only
        # concepts are independent of each other, no softmax across candidates, matching
        # SparseEmbeddingToConcept's sigmoid semantics

        # weight is exactly 1.0 in the forward pass, so this equals wei_p @ self.value, a pure
        # convex combination
        candidate_score = (
            wei_p @ self.value
        ) * weight  # shape: [B, T, C, P] @ [P] -> [B, T, C]

        k_dense = x.new_zeros(B, T, self.Kt)
        k_dense.scatter_add_(
            dim=-1, index=idx, src=candidate_score
        )  # 0 everywhere not selected
        return k_dense


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

    _NEG_MASK_VALUE = (
        -1e9
    )  # additive mask: large-but-finite, so an all-padded group softmaxes to
    # uniform (not NaN) -- true -inf can leak NaN into the backward pass even through a later
    # masked_fill/where, since softmax's own backward formula uses its (possibly all-NaN) output

    def __init__(
        self,
        in_embeddings,
        out_concepts,
        proto_token_ids,
        backbone,
        key_dim=None,
        top_k=5,
    ):
        super().__init__()
        d = in_embeddings
        Kt = out_concepts
        assert (
            proto_token_ids.shape[0] == Kt
        ), f"proto_token_ids has {proto_token_ids.shape[0]} concepts, expected {Kt}"
        assert tuple(proto_token_ids.shape[1:]) == (2, top_k), (
            f"proto_token_ids shape {tuple(proto_token_ids.shape[1:])} != (2, {top_k}) "
            "(axis 1: 0=negative, 1=positive -- see data.utils.load_lifted_token_prototypes)"
        )
        self.Kt = Kt
        self.top_k = top_k

        self.backbone = backbone  # shared with the main model; see module docstring
        self.register_buffer("proto_token_ids", proto_token_ids, persistent=False)

        self.register_buffer(
            "value", torch.tensor([-1.0] * top_k + [1.0] * top_k), persistent=False
        )
        self.proto_query = nn.Linear(d, d, bias=False)
        # cosine similarity between unit vectors concentrates tightly around 0 in high dimensions
        # (std ~= 1/sqrt(d)), so softmax over raw cosine sims is nearly uniform regardless of how
        # well-matched a candidate actually is. A learned temperature (CLIP's logit_scale trick)
        # lets softmax sharpen as needed while staying bounded -- clamped in forward() so it can
        # never grow back into the unbounded-magnitude problem cosine similarity was meant to fix.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, x, idx, weight):
        B, T = x.shape[:2]

        # only encode the distinct concepts some token actually picked, each at most once
        unique_idx, inverse = torch.unique(
            idx, return_inverse=True
        )  # unique_idx: [U]; inverse: [B, T, C]
        proto_ids_selected = self.proto_token_ids[
            unique_idx
        ]  # shape: [U, 2, top_k], single token ids

        # validity from the raw (possibly -1 = "no lifted token here") ids, before any clamping --
        # token id 0 is '<|endoftext|>', a real token that can legitimately be a lifted token
        # itself, so it must not be mistaken for padding (see data.utils.load_lifted_token_prototypes)
        valid_selected = proto_ids_selected >= 0  # shape: [U, 2, top_k]
        proto_ids_safe = proto_ids_selected.clamp(
            min=0
        )  # -1 -> 0, a valid (if dummy) embedding id;
        # only used for the lookup below -- its actual embedding never matters, since invalid
        # slots are masked out of the softmax before they can affect anything

        # each prototype is exactly one token: a length-1 sequence per prototype, so there's no
        # sentence to average-pool -- squeeze the trivial length-1 axis back out instead
        hidden = self.backbone(
            proto_ids_safe.reshape(-1, 1)
        )  # shape: [U*2*top_k, 1, d]
        hidden = hidden.squeeze(1).view(
            unique_idx.shape[0], 2, self.top_k, -1
        )  # [U, 2, top_k, d]
        # x and h are compared in the exact same learned space (same proto_query for both), so
        # nothing bounds the raw dot product's magnitude -- gradients on proto_query get pulled
        # from both its "query" and "key" role every step, reinforcing growth along its own
        # dominant direction. Cosine similarity removes that degree of freedom: only the angle
        # between x and h matters, not how large proto_query's weights have grown.
        key_selected = F.normalize(
            self.proto_query(hidden), dim=-1
        )  # shape: [U, 2, top_k, H]

        if x.device.type == "mps":
            # see PrototypePredictor.forward: MPS crashes when this gather's backward scatters
            # into a buffer shaped by U, which changes every step. Route through a fixed
            # [Kt, 2, top_k, H] buffer instead -- cheap since Kt is small (see nn.bottleneck).
            key_all = key_selected.new_zeros(
                self.Kt, 2, self.top_k, key_selected.shape[-1]
            )
            key_all[unique_idx] = key_selected
            key_cand = key_all[idx]  # shape: [B, T, C, 2, top_k, H]
            valid_all = proto_ids_selected.new_zeros(
                self.Kt, 2, self.top_k, dtype=torch.bool
            )
            valid_all[unique_idx] = valid_selected
            valid_cand = valid_all[idx]  # shape: [B, T, C, 2, top_k]
        else:
            key_cand = key_selected[inverse]  # shape: [B, T, C, 2, top_k, H]
            valid_cand = valid_selected[inverse]  # shape: [B, T, C, 2, top_k]

        q = F.normalize(self.proto_query(x), dim=-1)  # shape: [B, T, H]
        raw = torch.einsum(
            "bth,btcgkh->btcgk", q, key_cand
        )  # [B, T, C, 2, top_k], cosine similarity in [-1, 1]
        # clamp the log-value, not exp()'s result: exp() overflows to inf before a post-hoc clamp
        # could catch it, and inf's gradient combined with clamp's zero-grad region there
        # produces 0*inf = nan for logit_scale -- silently reintroducing the same failure this
        # temperature exists to prevent.
        scale = self.logit_scale.clamp(
            max=math.log(100)
        ).exp()  # bounded temperature, see __init__
        raw = raw * scale
        raw = raw.masked_fill(~valid_cand, self._NEG_MASK_VALUE)
        wei_p = raw.flatten(-2, -1).softmax(dim=-1)  # [B, T, C, 2*top_k]

        # convex combination of exactly +1 (positive) and -1 (negative), weighted by the
        # renormalized (max_pos, max_neg) pair; eps guards the (should-be-rare) case where a
        # concept has no valid tokens in either direction at all
        candidate_score = (
            wei_p @ self.value
        ) * weight  # shape: [B, T, C, 2*top_k] @ [2*top_k] -> [B, T, C]

        k_dense = x.new_zeros(B, T, self.Kt)
        k_dense.scatter_add_(
            dim=-1, index=idx, src=candidate_score
        )  # 0 everywhere not selected
        return k_dense


class PrototypeConceptEncoder(BaseConceptLayer):
    """Combines any Selector (LinearSelector, ProductKeySelector) with a PrototypePredictor
    into the activation()/embed()/forward()/ground_truth_embedding()/.K interface
    ConceptBottleneck expects. Swap `selector` without touching how candidates get scored.
    """

    def __init__(self, in_embeddings, out_concepts, selector, predictor):
        super().__init__(
            out_concepts=out_concepts, in_concepts=None, in_embeddings=in_embeddings
        )
        self.selector = selector
        self.predictor = predictor
        self.K = nn.Parameter(
            torch.randn(self.out_concepts_shape, self.in_embeddings_shape) * 0.02
        )  # value/reconstruction table, as in SparseEmbeddingToConcept

    def activation(self, x):
        idx, weight = self.selector(x)
        return self.predictor(x, idx, weight)

    def embed(self, k):
        return k @ self.K  # shape: [B, T, Kt] @ [Kt, d] -> [B, T, d]

    def forward(self, embeddings):
        alpha = self.activation(embeddings)
        return alpha, self.embed(alpha)

    def ground_truth_embedding(self, known_labels):
        return known_labels.float() @ self.K  # shape: [B, T, Kt] @ [Kt, d] -> [B, T, d]


class PrototypeCrossAttention(BaseConceptLayer):
    """Known-concept encoder: scores every token against every concept's prototypes directly,
    instead of through a Selector + Predictor (see module docstring for why).

    Concepts are processed in chunks of chunk_size. Each chunk is optionally wrapped in
    torch.utils.checkpoint, so backward recomputes it instead of keeping every chunk's
    [B, T, chunk_size, P, d] tensor in memory at once (the same memory-for-recompute trade
    FlashAttention makes).
    """

    def __init__(
        self,
        in_embeddings,
        out_concepts,
        proto_token_ids,
        backbone,
        chunk_size=4096,
        key_dim=None,
        top_k=None,
        use_checkpoint=True,
    ):
        super().__init__(
            out_concepts=out_concepts, in_concepts=None, in_embeddings=in_embeddings
        )
        d = self.in_embeddings_shape
        Kt = self.out_concepts_shape
        assert (
            proto_token_ids.shape[0] == Kt
        ), f"proto_token_ids has {proto_token_ids.shape[0]} concepts, expected {Kt}"
        self.Kt = Kt
        self.chunk_size = min(chunk_size, Kt)
        self.top_k = top_k
        self.use_checkpoint = use_checkpoint
        H = key_dim or min(d, 32)  # a scoring key doesn't need the full hidden width

        # only the cheap token embedding table is used here: this class touches all Kt concepts
        # every call and can't afford a full backbone pass per concept
        self.embedding_table = backbone.token_embedding_table
        proto_token_ids = proto_token_ids.reshape(
            Kt, -1, proto_token_ids.shape[-1]
        )  # [Kt, P, Tp]
        self.register_buffer("proto_token_ids", proto_token_ids, persistent=False)
        self.register_buffer(
            "value", _fixed_value_axis(proto_token_ids), persistent=False
        )

        self.proto_query = nn.Linear(d, H, bias=False)
        self.proto_key = nn.Linear(d, H, bias=False)

        self.K = nn.Parameter(
            torch.randn(Kt, d) * 0.02
        )  # value/reconstruction table, as in SparseEmbeddingToConcept

    def _score_chunk(self, q, proto_ids_chunk):
        """q: [B, T, H]; proto_ids_chunk: [chunk, P, Tp] -> [B, T, chunk] independent per-concept
        activation scores for this slice of the concept library."""
        hidden = self.embedding_table(proto_ids_chunk)  # shape: [chunk, P, Tp, d]
        proto_emb = _masked_mean_pool(hidden, proto_ids_chunk)  # [chunk, P, d]
        k = self.proto_key(proto_emb)  # shape: [chunk, P, H]
        wei = torch.einsum("bth,cph->btcp", q, k) / (
            k.shape[-1] ** 0.5
        )  # shape: [B, T, chunk, P]
        wei_p = wei.softmax(
            dim=-1
        )  # normalize over this concept's own P prototypes only
        return wei_p @ self.value  # shape: [B, T, chunk, P] @ [P] -> [B, T, chunk]

    def activation(self, x):
        q = self.proto_query(x)  # shape: [B, T, H]

        chunks = []
        for start in range(0, self.Kt, self.chunk_size):
            proto_ids_chunk = self.proto_token_ids[start : start + self.chunk_size]
            if self.use_checkpoint and self.training:
                chunk_scores = _checkpoint(
                    self._score_chunk, q, proto_ids_chunk, use_reentrant=False
                )
            else:
                chunk_scores = self._score_chunk(q, proto_ids_chunk)
            chunks.append(chunk_scores)

        k_dense = torch.cat(chunks, dim=-1)  # shape: [B, T, Kt]
        return sparsify_top_k(k_dense, self.top_k)

    def embed(self, k):
        return k @ self.K  # shape: [B, T, Kt] @ [Kt, d] -> [B, T, d]

    def forward(self, embeddings):
        alpha = self.activation(embeddings)
        return alpha, self.embed(alpha)

    def ground_truth_embedding(self, known_labels):
        return known_labels.float() @ self.K  # shape: [B, T, Kt] @ [Kt, d] -> [B, T, d]
