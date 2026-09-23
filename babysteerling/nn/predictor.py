import math

import torch
from torch import nn
from torch_concepts.nn import BaseConceptLayer


class LinearEmbeddingToConcept(BaseConceptLayer):
    """Projects the bottlenecked hidden state to vocabulary logits.
    """

    def __init__(self, in_embeddings, out_concepts, tie_weights=True, tied_embedding=None):
        super().__init__(
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
            in_concepts=None
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
        return self.head(k), self.head(u), self.head(epsilon)  # each: [B, T, d] -> [B, T, vocab_size]


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
                eps = layer.eps if layer.eps is not None else torch.finfo(summed.dtype).eps
                rms = (summed.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt().detach()  # frozen from the true sum
                scale = layer.weight / rms  # same divisor for every term -> distributes over the k+u+e split
                summed, k, u, e = layer(summed), k * scale, u * scale, e * scale
            else:
                summed, k, u, e = layer(summed), layer(k), layer(u), layer(e)
        return k, u, e


class LinearMemoryPredictor(BaseConceptLayer):
    def __init__(self, in_embeddings, out_concepts,
                 memory_embedding_dims=30, memory_size=5,
                 warmup_steps=200,
                 init_temperature=5.0, final_temperature=0.01,
                 anneal_steps=10_000, grad_clip_norm=1.0):

        super().__init__(
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
            in_concepts=None
        )
        d = self.in_embeddings_shape
        vocab_size = self.out_concepts_shape
        self.memory_embedding_dims = memory_embedding_dims
        self.memory_size = memory_size
        self.init_temperature = init_temperature
        self.final_temperature = final_temperature
        self.anneal_steps = anneal_steps
        self.warmup_steps = warmup_steps
        self.grad_clip_norm = grad_clip_norm
        self.head_type = "memory"

        self._selector_out_shape = (vocab_size, memory_size)
        self._selector_output_dim = torch.tensor(self._selector_out_shape).prod().item()

        self.memory = nn.Embedding(memory_size, d*vocab_size)
        self._init_memory_weight()
        self.selector = nn.Linear(in_embeddings, memory_size)

        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))  # saved/restored with state_dict

        if grad_clip_norm is not None:
            self._register_grad_clip_hooks(grad_clip_norm)

    def _register_grad_clip_hooks(self, max_norm):
        def _clip(grad):
            if not torch.isfinite(grad).all():
                return torch.zeros_like(grad)  # skip updating this param this step
            norm = grad.norm()
            return grad * (max_norm / (norm + 1e-6)) if norm > max_norm else grad

        self.memory.weight.register_hook(_clip)
        for p in self.selector.parameters():
            p.register_hook(_clip)

    @property
    def selection_temperature(self):
        effective_step = max(0, self._step.item() - self.warmup_steps)
        progress = min(1.0, effective_step / self.anneal_steps)
        # log-space interpolation: equal steps feel more even, since softmax's
        # sensitivity to temperature is highly nonlinear near the low end
        log_t = math.log(self.init_temperature) * (1 - progress) + math.log(self.final_temperature) * progress
        return math.exp(log_t)

    def _init_memory_weight(self):
        with torch.no_grad():
            w = self.memory.weight.view(self.memory_size, self.in_embeddings_shape, self.out_concepts_shape)
            for m in range(self.memory_size):
                nn.init.kaiming_uniform_(w[m], a=math.sqrt(5))  # matches nn.Linear default, per-slot

    def emb_to_mixing_probabilities(self, embeddings, advance_step=None):
        if advance_step is None:
            advance_step = self.training  # forward() path: advances by default
        mixing_logits = self.selector(embeddings).float().clamp(-30, 30)
        temp = self.selection_temperature
        if self.training:
            mixing_probs = torch.softmax(mixing_logits / temp, dim=-1)
            if advance_step:
                self._step += 1
        else:
            mixing_probs = torch.nn.functional.gumbel_softmax(mixing_logits, tau=temp, hard=False, dim=-1)
        return mixing_probs.to(embeddings.dtype)

    def mixing(self, x, mixing_probs):
        memory_weight = self.memory.weight.view(
            self.memory_size, self.in_embeddings_shape, self.out_concepts_shape
        )

        # memory-efficient path from before: never materializes [B,T,d,V]
        per_slot = torch.einsum('btd,mdv->btmv', x, memory_weight)
        return torch.einsum('btm,btmv->btv', mixing_probs, per_slot)

    def forward(self, embeddings):
        mixing_probs = self.emb_to_mixing_probabilities(embeddings)
        return self.mixing(embeddings, mixing_probs).clamp(-30, 30)

    def decompose(self, k, u, epsilon):
        """
        Exact decomposition of vocab_logits(k+u+epsilon) into three additive
        terms, using a single shared set of mixing weights.
        `embeddings` defaults to k+u+epsilon (i.e. the weights are derived
        from the true full embeddings) — override if you want the mixing
        computed from something else.
        """
        embeddings = k + u + epsilon
        mixing_probs = self.emb_to_mixing_probabilities(embeddings, advance_step=False)

        logits_k = self.mixing(k, mixing_probs)
        logits_u = self.mixing(u, mixing_probs)
        logits_eps = self.mixing(epsilon, mixing_probs)

        return logits_k, logits_u, logits_eps
