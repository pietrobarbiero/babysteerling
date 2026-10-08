from abc import ABC, abstractmethod
from dataclasses import field, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_concepts.nn import InferenceOutput


@dataclass
class LossOutput:
    """Structured return type for loss modules and composite loss orchestrators."""

    loss: torch.Tensor
    metrics_dict: dict[str, float] = field(default_factory=dict)

    def item(self) -> float:
        """Convenience method to get scalar value of total_loss."""
        return self.loss.item()

    def merge(self, other: "LossOutput") -> "LossOutput":
        """Merges another LossOutput into this one by adding total_losses and summing overlapping metric values."""
        merged_metrics = self.metrics_dict.copy()
        for k, v in other.metrics_dict.items():
            merged_metrics[k] = merged_metrics.get(k, 0.0) + v

        return LossOutput(
            loss=self.loss + other.loss,
            metrics_dict=merged_metrics,
        )

    def __add__(self, other: "LossOutput | torch.Tensor") -> "LossOutput":
        """Supports `output_a + output_b` operator syntax."""
        if isinstance(other, LossOutput):
            return self.merge(other)
        elif isinstance(other, torch.Tensor):
            return LossOutput(
                loss=self.loss + other,
                metrics_dict=self.metrics_dict.copy(),
            )
        return NotImplemented

    def __radd__(self, other: "LossOutput | torch.Tensor | int") -> "LossOutput":
        """Supports `sum([loss1, loss2])` starting from 0."""
        if other == 0:
            return self
        return self.__add__(other)


class Loss(ABC):
    """Abstract base class for modular loss functions."""

    @abstractmethod
    def forward(self, output: InferenceOutput, batch: dict) -> torch.Tensor:
        """Computes and returns a differentiable loss tensor."""
        pass


class CompositeLoss(nn.Module):
    """Meta-loss orchestrator that computes a weighted sum of Loss objects
    and collects diagnostic Metric values.
    """

    def __init__(
        self,
        losses: dict[str, tuple[nn.Module, float]],
        metrics: dict[str, nn.Module] | None = None,
    ):
        super().__init__()
        self.loss_modules = nn.ModuleDict(
            {name: module for name, (module, _) in losses.items()}
        )
        self.weights = {name: weight for name, (_, weight) in losses.items()}
        self.metric_modules = nn.ModuleDict(metrics or {})

    def forward(
        self,
        outputs: InferenceOutput | dict[str, InferenceOutput],
        batch: dict | None = None,
        active_losses: list[str] | None = None,
        active_metrics: list[str] | None = None,
    ) -> LossOutput:
        device = (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cpu")
        )
        total_loss = torch.tensor(0.0, device=device)
        metrics_dict: dict[str, float] = {}

        # 1. Compute weighted loss terms
        if active_losses:
            for name in active_losses:
                if name not in self.loss_modules:
                    continue

                loss_fn = self.loss_modules[name]
                weight = self.weights[name]
                if weight == 0.0:
                    continue

                loss_val = loss_fn(outputs, batch)

                total_loss = total_loss + (weight * loss_val)

                metrics_dict[f"{name}_loss"] = loss_val.detach().item()
                if weight != 1.0:
                    metrics_dict[f"{name}_loss_weighted"] = (
                        (weight * loss_val).detach().item()
                    )

        metrics_dict["total_loss"] = total_loss.detach().item()

        # 2. Update stateful metrics across batches (eval only)
        if not self.training and active_metrics:
            for name in active_metrics:
                if name in self.metric_modules:
                    self.metric_modules[name].update(outputs, batch)

        loss_output = LossOutput(loss=total_loss, metrics_dict=metrics_dict)

        return loss_output

    def compute_metrics(
        self, active_metrics: list[str] | None = None
    ) -> dict[str, float]:
        """Computes and resets accumulated epoch-level metrics (e.g., AUC, Accuracy)."""
        eval_metrics = {}
        for name, metric in self.metric_modules.items():
            if active_metrics is not None and name not in active_metrics:
                continue

            # torchmetrics or custom metrics returning a dict or tensor
            metric_val = metric.compute() if hasattr(metric, "compute") else None
            if isinstance(metric_val, dict):
                for k, v in metric_val.items():
                    eval_metrics[k] = v.item() if isinstance(v, torch.Tensor) else v
            elif isinstance(metric_val, torch.Tensor):
                eval_metrics[name] = metric_val.item()
            elif isinstance(metric_val, float):
                eval_metrics[name] = metric_val

            if hasattr(metric, "reset"):
                metric.reset()
        return eval_metrics

    def reset_metrics(self, active_metrics: list[str] | None = None):
        """Call this at epoch end/start to clear state accumulators."""
        for name, metric in self.metric_modules.items():
            if active_metrics is not None and name not in active_metrics:
                continue
            if hasattr(metric, "reset"):
                metric.reset()


class TokenLoss(nn.Module, Loss):
    """Token-level cross-entropy loss."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, output: InferenceOutput, batch: dict) -> torch.Tensor:
        logits = output.logits["next_token"]
        targets = batch["targets"]

        B, T, C = logits.shape
        return F.cross_entropy(
            logits.view(B * T, C),
            targets.view(B * T),
            ignore_index=self.ignore_index,
        )


class TokenResidualLoss(nn.Module, Loss):
    """Token-level cross-entropy loss."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, output: dict[str, InferenceOutput], batch: dict) -> torch.Tensor:
        f_logits = output["next_token"].logits["next_token"]
        g_out = output["next_token_residual"].value
        targets = batch["targets"]

        B, T, C = f_logits.shape
        logits_corrected = f_logits.detach() + g_out

        return F.cross_entropy(
            logits_corrected.reshape(B * T, C),
            targets.reshape(B * T),
            ignore_index=self.ignore_index,
        )


class DiffusionTokenLoss(nn.Module, Loss):
    """Masked-diffusion token loss with optional 1/p_mask ELBO importance weighting.

    Expects in batch:
        - "targets": LongTensor [B, T]
        - "mask": BoolTensor [B, T] indicating which positions were corrupted/masked
        - "mask_weights" (optional): Tensor [B, T] of sampling probabilities p_mask for weighting
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, output: InferenceOutput, batch: dict) -> torch.Tensor:
        logits = output.logits["next_token"]
        targets = batch.get("targets", None)
        mask = output.mask if hasattr(output, "mask") else None

        if targets is None or mask is None or mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        mask_weights = output.p_mask if hasattr(output, "p_mask") else None

        # Masked cross entropy: [n_masked, vocab] vs [n_masked]
        ce = F.cross_entropy(logits[mask], targets[mask], reduction="none")

        if mask_weights is None:
            return ce.mean()

        # Weighted mean: weight each position by 1 / p_mask
        w = 1.0 / mask_weights[mask]
        return (ce * w).sum() / w.sum()


class ConceptLoss(nn.Module, Loss):
    """Document-level soft-OR loss over predicted known concepts."""

    P_MAX = 1 - 1e-6  # ceiling on activations, so log1p(-p) can't hit log(0)

    def forward(self, output: InferenceOutput, batch: dict) -> torch.Tensor:
        logits = output.logits["concepts"]
        if logits is None:
            return torch.tensor(0.0, device=output.logits.device)

        doc_spans = batch.get("doc_spans", [])
        if not doc_spans:
            return torch.tensor(0.0, device=logits.device)

        n = logits.shape[-1]
        total = logits.new_zeros(())
        for batch_idx, tok_start, tok_end, concept_ids in doc_spans:
            # Span logits shape: [doc_len, n]
            z_span = logits[batch_idx, tok_start:tok_end, :]

            # log P(concept absent) = sum over tokens of -softplus(z_i)
            # strictly negative for all unconstrained logits z_i
            # log_p_none = -F.softplus(z_span).sum(dim=0)  # shape: [n]
            doc_len = z_span.shape[0]
            # Scaling softplus sum by length (or sqrt(length)) prevents long spans from dominating
            log_p_none = -F.softplus(z_span).sum(dim=0) / (doc_len**0.5)

            # Ensure log_p_none is strictly negative (<= -1e-7) so exp(log_p_none) < 1.0,
            # preventing log1p(-1.0) = log(0)
            log_p_none_safe = log_p_none.clamp(max=-1e-7)

            # log P(concept present somewhere in span) = log(1 - P(none))
            # Exactly matches your original `torch.log1p(-log_p_none.exp())`
            log_p_any = torch.log1p(-torch.exp(log_p_none_safe))  # shape: [n]

            y = logits.new_zeros(n)  # multi-hot ground truth label
            y[concept_ids] = 1.0

            # # BCE loss: -(y * log_p_any + (1 - y) * log_p_none)
            # doc_loss = -(y * log_p_any + (1 - y) * log_p_none).mean()

            pos_mask = y == 1.0
            neg_mask = y == 0.0

            # Average per positive concept
            pos_loss = (
                -log_p_any[pos_mask].mean()
                if pos_mask.any()
                else logits.new_tensor(0.0)
            )

            # Average per negative concept
            neg_loss = (
                -log_p_none[neg_mask].mean()
                if neg_mask.any()
                else logits.new_tensor(0.0)
            )

            # Each group contributes equally (1:1 weight) regardless of how sparse y is
            doc_loss = pos_loss + neg_loss

            total = total + doc_loss

        return total / len(
            doc_spans
        )  # average per-document loss, so batch size doesn't change the scale


class DiffusionConceptLoss(nn.Module, Loss):
    """Mask-aware document-level soft-OR concept loss for diffusion."""

    def forward(self, output: InferenceOutput, batch: dict) -> torch.Tensor:
        logits = output.logits["concepts"]
        if logits is None:
            return torch.tensor(0.0, device=output.logits.device)

        mask = getattr(output, "mask", None)
        doc_spans = batch.get("doc_spans", [])

        if logits is None or not doc_spans or mask is None or mask.sum() == 0:
            return torch.tensor(
                0.0, device=logits.device if logits is not None else torch.device("cpu")
            )

        n = logits.shape[-1]
        total = logits.new_zeros(())
        valid_spans = 0

        for batch_idx, tok_start, tok_end, concept_ids in doc_spans:
            # Mask slice for this span [doc_len]
            span_mask = mask[batch_idx, tok_start:tok_end]

            # If no tokens were masked in this span, skip (no diffusion loss step)
            if not span_mask.any():
                continue

            # Logits for MASKED tokens only [n_masked, n]
            z_span = logits[batch_idx, tok_start:tok_end, :][span_mask]

            # True soft-OR log P(absent) over corrupted/predicted positions.
            # Scaling by sqrt(n_masked) prevents spans with many masked positions from
            # dominating, mirroring ConceptLoss's length normalization.
            n_masked = z_span.shape[0]
            log_p_none = -F.softplus(z_span).sum(dim=0) / (n_masked**0.5)
            log_p_none_safe = log_p_none.clamp(max=-1e-7)
            log_p_any = torch.log1p(-torch.exp(log_p_none_safe))

            y = logits.new_zeros(n)
            y[concept_ids] = 1.0

            pos_mask = y == 1.0
            neg_mask = y == 0.0

            pos_loss = (
                -log_p_any[pos_mask].mean()
                if pos_mask.any()
                else logits.new_tensor(0.0)
            )
            neg_loss = (
                -log_p_none[neg_mask].mean()
                if neg_mask.any()
                else logits.new_tensor(0.0)
            )

            total = total + (pos_loss + neg_loss)
            valid_spans += 1

        return total / max(valid_spans, 1)


# Losses for unknown/residual models


class ReconstructionLoss(nn.Module, Loss):
    """MSE between residual representation and target."""

    def forward(
        self, output: dict[str, InferenceOutput], batch: dict = None
    ) -> torch.Tensor:
        u = output["unknown_out"].value["unknown_embedding"]
        if u is None:
            return torch.tensor(0.0, device=output["unknown_out"].value.device)

        u_target = output["unknown_target_embeddings_out"].value[
            "unknown_target_embedding"
        ]
        if u_target is None:
            return torch.tensor(0.0, device=output["unknown_out"].value.device)

        if not hasattr(output["unknown_out"], "mask"):
            return ((u - u_target) ** 2).mean()

        if output["unknown_out"].mask.sum() == 0:
            return torch.tensor(0.0, device=u.device)

        return (
            (u[output["unknown_out"].mask] - u_target[output["unknown_out"].mask]) ** 2
        ).mean()


class IndependenceLoss(nn.Module, Loss):
    """Penalizes correlation between k and u, so the unknown head doesn't just re-learn
    what the known head already captures.

    Only the unknown side gets gradient (k is detached): we want the unknown head to adapt
    to the known head, not the other way around, since the known head is anchored to human
    labels.
    """

    MAX_VALUE = (
        1.0  # ceiling, so a single large-covariance batch can't dominate the total loss
    )

    def forward(
        self, output: dict[str, InferenceOutput], batch: dict = None
    ) -> torch.Tensor:
        k = output["known_out"].value["known_embedding"]
        if k is None:
            return torch.tensor(0.0, device=output["known_out"].value.device)

        val = output["other_out"].value
        u = val["unknown_embedding"] if "residual" not in val else val["residual"]
        if u is None:
            return torch.tensor(0.0, device=output["known_out"].value.device)

        d = k.shape[-1]
        Hk = k.detach().reshape(
            -1, d
        )  # shape: [B, T, d] -> [B*T, d], flatten batch+time into one axis of "samples"
        Hu = u.reshape(-1, d)  # shape: [B, T, d] -> [B*T, d]
        num_tokens = Hk.shape[0]

        Phi = Hk - Hk.mean(
            dim=0, keepdim=True
        )  # shape: [B*T, d], center each feature across the batch
        Psi = Hu - Hu.mean(dim=0, keepdim=True)  # shape: [B*T, d]
        cross_cov = (
            Psi.t() @ Phi
        )  # shape: [d, B*T] @ [B*T, d] -> [d, d], empirical cross-covariance matrix
        hsic = (cross_cov**2).sum() / (
            d**2 * max(num_tokens - 1, 1)
        )  # normalized Frobenius norm^2
        return hsic.clamp(max=self.MAX_VALUE)


# Losses for interventions


class RespondLoss(nn.Module, Loss):
    """Eq. 31: Pushes the injected concept's activation alpha_c toward 1

    at its attributed positions, training the concept module to recognize
    the concept it was guided toward. Works in logit space like the other losses
    (-log(sigmoid(z)) == softplus(-z)), no explicit sigmoid needed.

    Expects in batch:
        - "random_intervention_ids": per-row concept index, broadcastable to [B, 1, 1].
        - "position_mask": BoolTensor [B, T] marking attributed positions.
    """

    def forward(self, output: InferenceOutput, batch: dict) -> torch.Tensor:
        logits = output.logits["concepts"]  # shape: [B, T, n]
        if logits is None:
            return torch.tensor(0.0, device=output.logits.device)

        position_mask = batch.get("position_mask", None)
        concept_ids = batch.get("random_intervention_ids", None)

        if position_mask is None or concept_ids is None or position_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        B, T, n = logits.shape
        idx = concept_ids.reshape(B, 1, 1).expand(B, T, 1).long()  # shape: [B, T, 1]
        z_c = logits.gather(-1, idx).squeeze(
            -1
        )  # shape: [B, T], each row's own intervened concept logit

        return F.softplus(-z_c[position_mask]).mean()


class ExpressLoss(nn.Module, Loss):
    """Eq. 32: Pushes the model's output token distribution toward the concept's lifted tokens at attributed positions.

    Expects in batch:
        - "lifted_token_ids": Collection/Tensor of token IDs for the target concept.
        - "position_mask": BoolTensor [B, T] marking attributed positions.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, output: InferenceOutput, batch: dict) -> torch.Tensor:
        logits = output.logits["next_token"]
        position_mask = batch.get("position_mask", None)
        lifted_token_ids = batch.get("lifted_tokens_intervention_ids", None)

        if (
            position_mask is None
            or lifted_token_ids is None
            or len(lifted_token_ids) == 0
            or position_mask.sum() == 0
        ):
            return torch.tensor(0.0, device=logits.device)

        probs = F.softmax(logits, dim=-1)  # shape: [B, T, vocab]
        mass = torch.take_along_dim(probs, lifted_token_ids.long(), dim=-1).sum(
            dim=-1
        )  # shape: [B, T]
        return -torch.log(mass[position_mask].clamp(self.eps, 1.0)).mean()
