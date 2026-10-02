from typing import Type

import torch
from torch import nn
from torch.distributions import OneHotCategorical
from torch.nn import functional as F
from torch_concepts import ConceptVariable
from torch_concepts.nn import ParametricCPD, BayesianNetwork, BaseInference

from .. import diffusion
from ..diffusion import build_block_causal_mask
from .lm import DLM
from ..loss import LossOutput
from ..nn.predictor import LinearEmbeddingToConcept


class BlockCausalDLM(DLM):
    """Block causal diffusion language model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int,
        num_heads: int,
        num_kv_heads: int,
        n_layers: int,
        dropout: float,
        loss_fn: nn.Module,
        inference: Type[BaseInference],

        # diffusion parameters
        mask_token_id: int,
        diff_block_len: int | None = None,

        inference_kwargs: dict | None = None,
        tie_weights: bool = True,
        **kwargs,
    ):

        assert diff_block_len is not None, "diff_block_len is required for the diffusion backbone"
        self.diff_block_len = diff_block_len
        self.mask_token_id = mask_token_id

        attn_mask = build_block_causal_mask(block_size, diff_block_len)

        super().__init__(vocab_size, block_size, n_embed, num_heads, num_kv_heads, n_layers, dropout, loss_fn, attn_mask, **kwargs)

        self.tie_weights = tie_weights
        tied_embedding = self.tokens_to_embedding.token_embedding_table.weight if tie_weights else None
        self._head = LinearEmbeddingToConcept(
            self.n_embed, self.vocab_size, tie_weights=tie_weights, tied_embedding=tied_embedding,
        )

        self.next_token = ConceptVariable("next_token", distribution=OneHotCategorical, size=1, members=[f"token_{i}" for i in range(vocab_size)])
        self.head_cpd = ParametricCPD(self.next_token, parametrization={"logits": self._head}, parents=[self.latent_var])

        self.pgm = BayesianNetwork(
            [self.input_var, self.latent_var, self.next_token],
            [self.input_cpd, self.latent_cpd, self.head_cpd]
        )
        self.inference = inference(self.pgm, **(inference_kwargs or {}))

    @property
    def head(self) -> nn.Module:
        return self._head

    @property
    def attn_mask(self) -> torch.Tensor | None:
        return self._attn_mask

    def step(self, batch: dict, *args, **kwargs) -> LossOutput:
        """Diffusive step: executes sequence corruption prior to forward pass."""
        corrupted_batch, mask, p_mask = diffusion.corrupt(batch["input_ids"], self.mask_token_id, self.diff_block_len)  # x_t, mask, p_mask: [batch_size, block_size]

        x = self.tokens_to_embedding(corrupted_batch)

        output = self.inference.query(
            query=["next_token"],
            evidence={"input": x},
        )

        output.corrupted_batch = corrupted_batch
        output.mask = mask
        output.p_mask = p_mask

        loss_output = self.loss_fn(output, batch, ["token"], ["token_accuracy"])

        return loss_output

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 50,
        gen_steps: int | None = None,
        **kwargs,
    ):
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        # set seq_len to self.block_size so x matches the 256x256 attention mask shape
        seq_len = self.block_size
        x = torch.full((batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device)

        # Clean integer division since seq_len (self.block_size) is guaranteed divisible by diff_block_len
        num_blocks = seq_len // self.diff_block_len
        steps_per_block = None if gen_steps is None else max(1, gen_steps // num_blocks)

        for block in range(num_blocks):
            lo = block * self.diff_block_len
            hi = lo + self.diff_block_len

            step = 0
            while True:
                still_masked = (x[0, lo:hi] == self.mask_token_id).nonzero(as_tuple=True)[0]
                if len(still_masked) == 0:
                    break

                x_emb = self.tokens_to_embedding(x)
                out = self.inference.query(query=["next_token"], evidence={"input": x_emb})
                block_logits = out.logits["next_token"][0, lo:hi] / temperature
                block_logits[:, self.mask_token_id] = float("-inf")

                probs = F.softmax(block_logits, dim=-1)
                if top_k is not None:
                    v, _ = torch.topk(block_logits, min(top_k, block_logits.size(-1)), dim=-1)
                    probs = torch.where(block_logits < v[:, [-1]], torch.zeros_like(probs), probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                confidence = probs[still_masked].max(dim=-1).values
                n_commit = 1 if steps_per_block is None else (
                    -(-len(still_masked) // max(1, steps_per_block - step))
                )
                pos = still_masked[confidence.topk(n_commit).indices]
                x[0, lo + pos] = torch.multinomial(probs[pos], num_samples=1).squeeze(-1)
                step += 1

        if was_training:
            self.train()

        # Slice output back to requested max_new_tokens before returning list
        return x[0, :max_new_tokens].tolist()
