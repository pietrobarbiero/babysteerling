from functools import reduce
from operator import sub
from typing import Type, Union

import torch
from torch import nn as nn
from torch.distributions import Bernoulli, OneHotCategorical
from torch.nn import functional as F
from torch_concepts import Annotations, ConceptVariable, EmbeddingVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import BaseInference, ParametricCPD, BayesianNetwork

from .lm import ILM
from .dlm import BlockCausalDLM
from .. import diffusion, steering
from ..loss import LossOutput
from ..nn.encoder import ResidualModule, SparseEmbeddingToConcept, ConceptToLowRankEmbeddings, CallableModule


class SteerlingDLM(BlockCausalDLM, ILM):
    """Language model with Steerling-style bottleneck and block causal diffusion."""

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
        out_concepts: Union[int, Annotations],

        diff_block_len: int | None = None,

        # bottleneck parameters
        unknown_ratio: float = 3,
        p_epsilon: float = 0.1,
        unknown_rank: int = 32,
        top_k_known: int = None,
        top_k_unknown: int = None,

        tie_weights: bool = True,

        inference_kwargs: dict | None = None,

        steering_every_n_steps: int = 10,
        inj_layer: int = 1,
        tau: float = 4.0,
        **kwargs
    ):
        super().__init__(vocab_size, block_size, n_embed, num_heads, num_kv_heads, n_layers, dropout, loss_fn, inference, mask_token_id, diff_block_len, inference_kwargs, tie_weights, **kwargs)
        if isinstance(out_concepts, int):
            out_concepts = [f"concept_{i}" for i in range(out_concepts)]
            n_concepts = len(out_concepts)
        else:
            n_concepts = len(out_concepts)

        self.n_unknown_concepts = int(unknown_ratio * n_concepts)
        out_unknown_concepts = [f"unknown_{i}" for i in range(self.n_unknown_concepts)]

        self.steering_every_n_steps = steering_every_n_steps
        self.inj_layer = inj_layer
        self.tau = tau

        self._bottleneck = nn.ModuleDict({
            "known": SparseEmbeddingToConcept(n_embed, n_concepts, top_k=top_k_known),
            "unknown": SparseEmbeddingToConcept(n_embed, self.n_unknown_concepts, top_k=top_k_unknown),

            "known_embeddings": ConceptToLowRankEmbeddings(n_concepts, n_embed),
            "unknown_embeddings": ConceptToLowRankEmbeddings(self.n_unknown_concepts, n_embed, rank=unknown_rank),
            "unknown_target_embeddings": CallableModule(lambda c: torch.sub(*c.chunk(2, dim=-1))),
            "residual_embeddings": CallableModule(lambda c: reduce(sub, c.chunk(3, dim=-1))),

            "residual": ResidualModule(p_epsilon),

            "reconstruction": CallableModule(lambda c: sum(c.chunk(3, dim=-1))),
        })

        # Define variables
        self.known_concepts = ConceptVariable("concepts", distribution=Bernoulli, size=1, members=list(out_concepts))
        self.unknown_concepts = ConceptVariable("unknown", distribution=Bernoulli, size=1, members=list(out_unknown_concepts))
        self.known_embeddings = EmbeddingVariable("known_embedding", distribution=Delta, size=n_embed)
        self.unknown_embeddings = EmbeddingVariable("unknown_embedding", distribution=Delta, size=n_embed)
        self.unknown_target_embeddings = EmbeddingVariable("unknown_target_embedding", distribution=Delta, size=n_embed)
        self.residual_embeddings = EmbeddingVariable("residual_embedding", distribution=Delta, size=n_embed)
        self.residual = EmbeddingVariable("residual", distribution=Delta, size=n_embed)
        self.reconstruction = EmbeddingVariable("reconstruction", distribution=Delta, size=n_embed)
        self.next_token = ConceptVariable("next_token", distribution=OneHotCategorical, size=1, members=[f"token_{i}" for i in range(vocab_size)])

        # Define CPDs
        self.concepts_cpd = ParametricCPD(self.known_concepts, parametrization={"logits": self._bottleneck["known"]}, parents=[self.latent_var])
        self.unknown_cpd = ParametricCPD(self.unknown_concepts, parametrization={"logits": self._bottleneck["unknown"]}, parents=[self.latent_var])
        self.known_embeddings_cpd = ParametricCPD(self.known_embeddings, parametrization=self._bottleneck["known_embeddings"], parents=[self.known_concepts])
        self.unknown_embeddings_cpd = ParametricCPD(self.unknown_embeddings, parametrization=self._bottleneck["unknown_embeddings"], parents=[self.unknown_concepts])
        self.unknown_target_embeddings_cpd = ParametricCPD(self.unknown_target_embeddings, parametrization=self._bottleneck["unknown_target_embeddings"], parents=[self.latent_var, self.known_embeddings])
        self.residual_embeddings_cpd = ParametricCPD(self.residual_embeddings, parametrization=self._bottleneck["residual_embeddings"], parents=[self.latent_var, self.known_embeddings, self.unknown_embeddings])
        self.residual_cpd = ParametricCPD(self.residual, parametrization=self._bottleneck["residual"], parents=[self.residual_embeddings])
        self.reconstruction_cpd = ParametricCPD(self.reconstruction, parametrization=self._bottleneck["reconstruction"], parents=[self.known_embeddings, self.unknown_embeddings, self.residual])
        self.head_cpd = ParametricCPD(self.next_token, parametrization={"logits": self._head}, parents=[self.reconstruction])

        self.pgm = BayesianNetwork(
            [self.input_var, self.latent_var, self.known_concepts, self.unknown_concepts, self.known_embeddings, self.unknown_embeddings, self.unknown_target_embeddings, self.residual_embeddings, self.residual, self.reconstruction, self.next_token],
            [self.input_cpd, self.latent_cpd, self.concepts_cpd, self.unknown_cpd, self.known_embeddings_cpd, self.unknown_embeddings_cpd, self.unknown_target_embeddings_cpd, self.residual_embeddings_cpd, self.residual_cpd, self.reconstruction_cpd, self.head_cpd]
        )
        self.inference = inference(self.pgm, **(inference_kwargs or {}))


    @property
    def bottleneck(self) -> nn.Module:
        return self._bottleneck

    def step(self, batch: dict, *args, **kwargs) -> LossOutput:
        corrupted_batch, mask, p_mask = diffusion.corrupt(batch["input_ids"], self.mask_token_id, self.diff_block_len)  # x_t, mask, p_mask: [batch_size, block_size]

        x = self.tokens_to_embedding(corrupted_batch)

        latent_out = self.inference.query(
            query=["latent"],
            evidence={"input": x},
        )
        known_out = self.inference.query(
            query=["concepts", "known_embedding"],
            evidence={"latent": latent_out.value.tensor},
        )
        unknown_out = self.inference.query(
            query=["unknown", "unknown_embedding"],
            evidence={"latent": latent_out.value.tensor.detach()},
        )

        k_gt = known_out.value.tensor
        unknown_target_embeddings_out = None
        if "known_labels" in batch:
            k_gt = self.bottleneck["known_embeddings"].ground_truth_embedding(batch["known_labels"])  # shape: [B, T, d]
            unknown_target_embeddings_out = self.inference.query(
                query=["unknown_target_embedding"],
                evidence={"latent": latent_out.value.tensor, "known_embedding": k_gt},
            )

        head_out = self.inference.query(
            query=["residual", "reconstruction", "next_token"],
            evidence={"latent": latent_out.value.tensor, "known_embedding": k_gt, "unknown_embedding": unknown_out.value.tensor},
        )
        head_out.corrupted_batch = corrupted_batch
        head_out.mask = mask
        head_out.p_mask = p_mask
        unknown_out.mask = mask
        known_out.mask = mask

        total_loss = self.loss_fn(head_out, batch, ["token"], ["token_accuracy"])
        total_loss += self.loss_fn(known_out, batch, ["concept"], ["concept_auc"])
        total_loss += self.loss_fn(
            {"unknown_out": unknown_out, "unknown_target_embeddings_out": unknown_target_embeddings_out},
            None,
            ["reconstruction"]
        )
        total_loss += self.loss_fn(
            {"known_out": known_out, "other_out": unknown_out},
            None,
            ["independence"]
        )
        total_loss += self.loss_fn(
            {"known_out": known_out, "other_out": head_out},
            None,
            ["independence"]
        )

        if self.training and kwargs["step"] % self.steering_every_n_steps == 0 and "random_intervention_ids" in batch:
            concept_ids = batch["random_intervention_ids"].reshape(-1).long()  # shape: [B]
            known_embeddings = self.bottleneck["known_embeddings"]
            table = known_embeddings.K if known_embeddings.rank is None else known_embeddings.A @ known_embeddings.B
            directions = F.normalize(table[concept_ids], dim=-1)  # shape: [B, D], each row's own concept direction
            gamma = steering.calibrate_gamma(directions, self._head, self.tau)  # shape: [B]

            with steering.injected_at(self, directions, gamma, batch["position_mask"], self.inj_layer):
                steer_x = self.tokens_to_embedding(corrupted_batch)
                steer_out = self.inference.query(
                    query=["concepts", "next_token"],
                    evidence={"input": steer_x},
                )

            total_loss += self.loss_fn(steer_out, batch, ["steering"])
            total_loss += self.loss_fn(steer_out, batch, ["respond"])

        return total_loss
