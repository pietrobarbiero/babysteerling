from typing import Callable

import torch
import torch.nn as nn
from torch.nn import Module

from torch_concepts.nn import BaseConceptLayer


def sparsify_top_k(activations, k):
    """Zeros out every activation except the top-k per token. Off by default.

    Forces each token to explain itself with a few active concepts instead of a dense mixture,
    closer to how a person would describe a piece of text.
    """
    if k is None or k >= activations.shape[-1]:
        return activations
    top_vals, top_idx = torch.topk(activations, k, dim=-1)  # shape: [..., n] -> [..., k] (values and their indices)
    sparse = torch.zeros_like(activations)
    sparse.scatter_(-1, top_idx, top_vals)  # write the top-k values back into their original positions, rest stay 0
    return sparse


class SparseEmbeddingToConcept(BaseConceptLayer):
    """Predicts concept activations alpha from input token embeddings."""

    def __init__(self, in_embeddings, out_concepts, top_k=None, logit_clamp=15.0):
        super().__init__(
            out_concepts=out_concepts,
            in_concepts=None,
            in_embeddings=in_embeddings,
        )
        m = self.out_concepts_shape
        d = self.in_embeddings_shape
        self.predictor = nn.Linear(d, m)
        nn.init.normal_(self.predictor.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.predictor.bias)
        self.logit_clamp = logit_clamp
        self.top_k = top_k

    def activation(self, embeddings):
        logits = self.predictor(embeddings)
        if self.logit_clamp is not None:
            logits = logits.clamp(-self.logit_clamp, self.logit_clamp)
        # Return raw logits, not probabilities: the inference engine's Bernoulli CPD
        # applies the sigmoid itself when turning these logits into a concept value.
        return sparsify_top_k(logits, self.top_k)

    def forward(self, embeddings):
        return self.activation(embeddings)


class ConceptToLowRankEmbeddings(Module):
    """Embeds concept activations alpha back into token embedding space."""

    def __init__(self, in_concepts, out_embeddings, rank=None):
        super().__init__()
        m = in_concepts
        d = out_embeddings
        self.rank = rank
        if rank is None:
            self.K = nn.Parameter(torch.randn(m, d) * 0.02)
        else:
            self.A = nn.Parameter(torch.randn(m, rank) * 0.02)
            self.B = nn.Parameter(torch.randn(rank, d) * 0.02)

    def embed(self, u):
        if self.rank is None:
            return u @ self.K
        return (u @ self.A) @ self.B

    def forward(self, alpha):
        return self.embed(alpha)

    def ground_truth_embedding(self, known_labels):
        if self.rank is None:
            return known_labels.float() @ self.K
        return (known_labels.float() @ self.A) @ self.B


class ResidualModule(nn.Module):
    """epsilon = h - k - u: whatever the two concept heads don't reconstruct.

    Dropout on epsilon discourages the model from routing information through this
    uninterpretable channel just because it's easier than using a concept. It only bites when the
    whole residual is gone, since h_bar is then k + u and those two have to carry the
    sequence on their own. Per-element dropout doesn't achieve that: it zeroes 10% of individual
    numbers and rescales the rest by 1/(1-p), so h_bar stays close to h on every sequence. Dropping
    the residual for 10% of *sequences* instead, all or nothing.
    """

    def __init__(self, p_epsilon=0.1):
        super().__init__()
        self.p_epsilon = p_epsilon

    def forward(self, epsilon):
        # epsilon = h - k - u  # shape: [B, T, d], same shape as h
        if self.training and self.p_epsilon > 0:
            keep = (torch.rand(epsilon.shape[0], 1, 1, device=epsilon.device)
                    >= self.p_epsilon).to(epsilon.dtype)  # shape: [B, 1, 1], one draw per sequence
            epsilon = epsilon * keep
        return epsilon


class CallableModule(Module):
    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward(self, input):
        output = self.func(input)
        return output
