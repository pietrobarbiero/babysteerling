import torch
from torch import nn
from torch_concepts.nn import BaseConceptLayer


class LinearEmbeddingToConcept(BaseConceptLayer):
    """Projects the bottlenecked hidden state to vocabulary logits."""

    def __init__(
        self, in_embeddings, out_concepts, tie_weights=True, tied_embedding=None
    ):
        super().__init__(
            in_embeddings=in_embeddings, out_concepts=out_concepts, in_concepts=None
        )
        d = self.in_embeddings_shape
        vocab_size = self.out_concepts_shape
        self.head_type = "linear"
        self.head = nn.Linear(d, vocab_size, bias=False)
        if tie_weights and tied_embedding is not None:
            self.head.weight = tied_embedding  # share the tensor, not a copy

    def forward(self, embeddings):
        return self.head(embeddings)  # shape: [B, T, d] -> [B, T, vocab_size]

    def decompose(self, k, u, epsilon):
        """Split logits into known/unknown/residual contributions.

        Exact (sums to forward(k + u + epsilon)) when head_type="linear", since the head is then a
        single linear map with no bias. Only approximate for head_type="mlp".
        """
        return (
            self.head(k),
            self.head(u),
            self.head(epsilon),
        )  # each: [B, T, d] -> [B, T, vocab_size]


class ReluEmbeddingToConcepts(LinearEmbeddingToConcept):
    """Same as LinearEmbeddingToConcept, but with a small ReLU MLP head instead of one linear
    layer, pre-normed with RMSNorm (no bias, no mean-centering -- see decompose() for why that
    matters) so activation scale can't compound unboundedly across the head's three stacked
    layers the way it could with nothing between them. Locally linear: for a fixed input, ReLU
    is just a fixed 0/1 mask and RMSNorm is just a division by a fixed per-instance scalar, so
    the whole head behaves like a linear map for that input (see decompose() below).
    """

    def __init__(self, in_embeddings, out_concepts, mlp_hidden=None):
        super().__init__(
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )
        d = self.in_embeddings_shape
        vocab_size = self.out_concepts_shape
        self.head_type = "non-linear"
        mlp_hidden = mlp_hidden or d
        # deeper, nonlinear head; weight tying isn't meaningful here since the final
        # layer's input space isn't the embedding space
        self.head = nn.Sequential(
            nn.RMSNorm(d),
            nn.Linear(d, mlp_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.RMSNorm(mlp_hidden),
            nn.Linear(mlp_hidden, mlp_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.RMSNorm(mlp_hidden),
            nn.Linear(mlp_hidden, vocab_size, bias=False),
        )

    def decompose(self, k, u, epsilon):
        """Split logits into known/unknown/residual contributions.

        The head has no biases, so each Linear layer is additive: layer(k) + layer(u) +
        layer(e) == layer(k + u + e). ReLU, for a fixed input, is just multiplying by a fixed
        0/1 mask; RMSNorm, for a fixed input, is just dividing by a fixed per-instance scalar
        (RMSNorm has no mean-centering or bias, unlike LayerNorm, so it distributes over a sum
        the same way a bias-free Linear does -- LayerNorm's mean-subtraction wouldn't: applied
        separately to k, u, e it would subtract the mean three times instead of once). Both
        statistics (the ReLU mask, RMSNorm's RMS) are computed once from the true sum
        z = k + u + e and applied identically to all three terms, so they still add up exactly
        to forward(k + u + epsilon).
        """
        summed = k + u + epsilon
        e = epsilon
        for layer in self.head:
            if isinstance(layer, nn.ReLU):
                mask = (summed > 0).to(summed.dtype)
                summed, k, u, e = summed * mask, k * mask, u * mask, e * mask
            elif isinstance(layer, nn.RMSNorm):
                eps = (
                    layer.eps
                    if layer.eps is not None
                    else torch.finfo(summed.dtype).eps
                )
                rms = (
                    (summed.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt().detach()
                )  # frozen from the true sum
                scale = (
                    layer.weight / rms
                )  # same divisor for every term -> distributes over the k+u+e split
                summed, k, u, e = layer(summed), k * scale, u * scale, e * scale
            else:
                summed, k, u, e = layer(summed), layer(k), layer(u), layer(e)
        return k, u, e
