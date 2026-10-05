import os
from typing import Any

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from .data.utils import overlapping_docs, filter_concepts_by_lifted_tokens
from .steering import sample_steering_target


class ConceptDataset(Dataset):
    """Memory-efficient Dataset yielding single window dicts.

    Operates strictly per-window (not per-batch) using binary-search overlap logic.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        doc_records: list[dict],
        doc_starts: list[int],
        block_size: int,
        n_concepts: int,
        lifted_tokens: dict[int, Any],
        start_idx: int = 0,
        end_idx: int | None = None,
    ):
        self.tokens = tokens  # Memory-mapped 1D LongTensor
        self.doc_records = doc_records
        self.doc_starts = doc_starts
        self.block_size = block_size
        self.n_concepts = n_concepts
        self.lifted_tokens = lifted_tokens

        # Define token index bounds for train/val splits
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(tokens)

    def __len__(self) -> int:
        available_tokens = self.end_idx - self.start_idx
        return max(0, (available_tokens - self.block_size - 1) // self.block_size)

    def __getitem__(self, idx: int) -> dict:
        # Compute absolute window offsets in the token stream
        window_start = self.start_idx + (idx * self.block_size)
        window_end = window_start + self.block_size

        # 1. Slice tokens directly from disk via mmap (zero full-corpus RAM loading)
        x = self.tokens[window_start:window_end].clone().long()
        y = self.tokens[window_start + 1 : window_end + 1].clone().long()

        # 2. Per-window binary search for document spans & dense multi-hot labels
        doc_spans = []
        known_labels = torch.zeros(
            self.block_size, self.n_concepts, dtype=torch.float32
        )

        for tok_start, tok_end, concept_ids in overlapping_docs(
            self.doc_records, self.doc_starts, window_start, window_end
        ):
            doc_spans.append((tok_start, tok_end, concept_ids))
            if concept_ids:
                known_labels[tok_start:tok_end, concept_ids] = 1.0

        random_intervention_id = sample_steering_target(doc_spans, self.lifted_tokens)

        doc_mask = torch.zeros_like(x, dtype=torch.bool)
        for tok_start, tok_end, concept_ids in doc_spans:
            if random_intervention_id in concept_ids:
                doc_mask[tok_start:tok_end] = True

        lifted = torch.as_tensor(list(self.lifted_tokens), device=x.device)
        position_mask = doc_mask & torch.isin(x, lifted)

        lifted_tokens_intervention_id = self.lifted_tokens.get(
            random_intervention_id, []
        )

        return {
            "input_ids": x,
            "targets": y,
            "doc_spans": doc_spans,
            "known_labels": known_labels,
            "random_intervention_id": torch.IntTensor(
                [random_intervention_id]
            ).unsqueeze(0),
            "position_mask": position_mask,
            "lifted_tokens_intervention_id": torch.IntTensor(
                [lifted_tokens_intervention_id]
            ),
        }


def concept_collate_fn(batch: list[dict]) -> dict:
    """Collates individual single-window items into structured batch dicts."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    targets = torch.stack([item["targets"] for item in batch])
    known_labels = torch.stack([item["known_labels"] for item in batch])
    random_intervention_ids = torch.stack(
        [torch.tensor(item["random_intervention_id"]) for item in batch]
    )
    position_mask = torch.stack([item["position_mask"] for item in batch])
    lifted_tokens_intervention_ids = torch.stack(
        [item["lifted_tokens_intervention_id"] for item in batch]
    )

    # Merge per-window doc_spans, adding the batch index b to each tuple
    batch_doc_spans = []
    for b, item in enumerate(batch):
        for tok_start, tok_end, concept_ids in item["doc_spans"]:
            batch_doc_spans.append((b, tok_start, tok_end, concept_ids))

    return {
        "input_ids": input_ids,  # [B, block_size]
        "targets": targets,  # [B, block_size]
        "known_labels": known_labels,  # [B, block_size, n_concepts]
        "doc_spans": batch_doc_spans,  # list of (b, tok_start, tok_end, concept_ids)
        "random_intervention_ids": random_intervention_ids,  # [B]
        "position_mask": position_mask,  # [B, block_size]
        "lifted_tokens_intervention_ids": lifted_tokens_intervention_ids,  # [B, lifted_tokens_of_intervened_concept]
    }


class ConceptDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for memory-efficient concept LM data loading.

    Handles memory-mapped token streaming, concept filtering, train/val splits,
    and automatic DataLoader construction.
    """

    def __init__(
        self,
        data_dir: str,
        block_size: int = 1024,
        batch_size: int = 32,
        min_lifted_tokens: int = 5,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_val_split: float = 0.9,
    ):
        super().__init__()
        self.save_hyperparameters()

        # State set during setup()
        self.n_concepts: int | None = None
        self.concepts: list[dict] | None = None
        self.lifted_tokens: dict | dict[int, Any] | None = None
        self.train_dataset: ConceptDataset | None = None
        self.val_dataset: ConceptDataset | None = None

    def setup(self, stage: str | None = None):
        """Prepares datasets for training/validation.

        Executes on every GPU process in distributed training.
        """
        if self.train_dataset is not None and self.val_dataset is not None:
            return  # Already initialized

        # 1. Open token stream with mmap=True so it's NOT loaded fully into RAM
        tokens_path = os.path.join(self.hparams.data_dir, "steerling_tokens.pt")
        tokens = torch.load(tokens_path, mmap=True)

        # 2. Load concept metadata & apply in-memory filtering
        doc_records = torch.load(
            os.path.join(self.hparams.data_dir, "steerling_concepts.pt")
        )
        with open(os.path.join(self.hparams.data_dir, "concepts.json")) as f:
            import json

            raw_n_concepts = len(json.load(f))

        doc_records, self.n_concepts, self.concepts, lifted_tokens = (
            filter_concepts_by_lifted_tokens(
                self.hparams.data_dir,
                doc_records,
                raw_n_concepts,
                min_lifted_tokens=self.hparams.min_lifted_tokens,
            )
        )

        doc_starts = [d["start"] for d in doc_records]
        n_train = int(self.hparams.train_val_split * len(tokens))

        # 3. Instantiate Train/Val Datasets sharing the memory-mapped token stream
        if stage in ("fit", None):
            self.train_dataset = ConceptDataset(
                tokens=tokens,
                doc_records=doc_records,
                doc_starts=doc_starts,
                block_size=self.hparams.block_size,
                n_concepts=self.n_concepts,
                lifted_tokens=lifted_tokens,
                start_idx=0,
                end_idx=n_train,
            )

            self.val_dataset = ConceptDataset(
                tokens=tokens,
                doc_records=doc_records,
                doc_starts=doc_starts,
                block_size=self.hparams.block_size,
                n_concepts=self.n_concepts,
                lifted_tokens=lifted_tokens,
                start_idx=n_train,
                end_idx=len(tokens),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=concept_collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=concept_collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )
