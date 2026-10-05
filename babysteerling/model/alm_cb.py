from typing import Type, Union

import torch
from torch import nn as nn
from torch.distributions import OneHotCategorical, Bernoulli
from torch.nn import Linear, Sequential
from torch_concepts import ConceptVariable, Annotations
from torch_concepts.nn import (
    ParametricCPD,
    BayesianNetwork,
    BaseInference,
    LinearEmbeddingToConcept,
)

from .alm import ALM
from .lm import ILM
from ..loss import LossOutput


class ConceptBottleneckALM(ALM, ILM):
    """Autoregressive language model with vanilla concept bottleneck."""

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
        # bottleneck parameters
        out_concepts: Union[int, Annotations],
        tie_weights: bool = True,
        inference_kwargs: dict | None = None,
        steering_every_n_steps: int = 10,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            block_size,
            n_embed,
            num_heads,
            num_kv_heads,
            n_layers,
            dropout,
            loss_fn,
            inference,
            tie_weights,
            inference_kwargs,
            **kwargs,
        )
        if isinstance(out_concepts, int):
            out_concepts = [f"concept_{i}" for i in range(out_concepts)]
            n_concepts = len(out_concepts)
        else:
            n_concepts = len(out_concepts)

        self.steering_every_n_steps = steering_every_n_steps

        self._bottleneck = nn.ModuleDict(
            {"known": LinearEmbeddingToConcept(n_embed, n_concepts)}
        )
        self._head = Sequential(
            Linear(n_concepts, n_embed),
            nn.GELU(),
            Linear(n_embed, n_embed),
            nn.GELU(),
            Linear(n_embed, vocab_size),
        )
        mask = None
        self.register_buffer("_attn_mask", mask, persistent=False)

        self.concepts = ConceptVariable(
            "concepts", distribution=Bernoulli, size=1, members=list(out_concepts)
        )
        self.next_token = ConceptVariable(
            "next_token",
            distribution=OneHotCategorical,
            size=1,
            members=[f"token_{i}" for i in range(vocab_size)],
        )

        self.concepts_cpd = ParametricCPD(
            self.concepts,
            parametrization={"logits": self._bottleneck["known"]},
            parents=[self.latent_var],
        )
        self.head_cpd = ParametricCPD(
            self.next_token,
            parametrization={"logits": self._head},
            parents=[self.concepts],
        )

        self.pgm = BayesianNetwork(
            [self.input_var, self.latent_var, self.concepts, self.next_token],
            [self.input_cpd, self.latent_cpd, self.concepts_cpd, self.head_cpd],
        )
        self.inference = inference(self.pgm, **(inference_kwargs or {}))

    @property
    def head(self) -> nn.Module:
        return self._head

    @property
    def bottleneck(self) -> nn.ModuleDict:
        return self._bottleneck

    def step(self, batch: dict, *args, **kwargs) -> LossOutput:
        x = self.tokens_to_embedding(batch["input_ids"])

        if self.training:
            output = self.inference.query(
                query=["concepts"],
                evidence={"input": x},
            )
            total_loss = self.loss_fn(output, batch, ["concept"], ["concept_auc"])

            # teacher forcing: use known labels to predict next token
            output = self.inference.query(
                query=["next_token"],
                evidence={"concepts": batch["known_labels"]},
            )
            total_loss += self.loss_fn(output, batch, ["token"], ["token_accuracy"])

        else:
            output = self.inference.query(
                query=["next_token", "concepts"],
                evidence={"input": x},
            )
            total_loss = self.loss_fn(
                output, batch, ["token", "concept"], ["token_accuracy", "concept_auc"]
            )

        if (
            self.training
            and kwargs["step"] % self.steering_every_n_steps == 0
            and "random_intervention_ids" in batch
        ):
            B, T, C = batch["known_labels"].shape
            intervention_mask = batch["random_intervention_ids"] == torch.arange(
                C, device=batch["random_intervention_ids"].device
            )

            intervened_labels = torch.where(
                intervention_mask, 1.0, batch["known_labels"]
            )

            output = self.inference.query(
                query=["next_token"],
                evidence={"concepts": intervened_labels},
            )
            total_loss += self.loss_fn(output, batch, ["steering"])

        return total_loss
