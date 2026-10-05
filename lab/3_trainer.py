import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from torch_concepts.nn import DeterministicInference

from babysteerling.loader import ConceptDataModule
from babysteerling.loss import CompositeLoss, ConceptLoss, TokenLoss
from babysteerling.metric import TokenAccuracy, ConceptAUC
from babysteerling.model.alm_cb import ConceptBottleneckALM
from babysteerling.trainer import LightningLM


def main():
    # 1. Initialize and set up the DataModule
    dm = ConceptDataModule(
        data_dir="./data",
        block_size=1024,
        batch_size=32,
        num_workers=4,
    )

    # 2. Model & Loss Hyperparameters
    vocab_size = 2000
    block_size = dm.hparams.block_size
    n_embed = 128
    num_heads = 4
    num_kv_heads = 2
    n_layers = 4
    dropout = 0.2

    # 3. Instantiate CompositeLoss & Metrics
    loss_fn = CompositeLoss(
        losses={
            "token": (TokenLoss(), 1.0),
            "concept": (ConceptLoss(), 1.0),
        },
        metrics={
            "token_accuracy": TokenAccuracy(),
            "concept_auc": ConceptAUC(),
        },
    )

    # 4. Instantiate Underlying PyTorch Model
    # Note: dm.n_concepts is available after initializing DataModule or calling dm.setup("fit")
    dm.setup("fit")
    inference_type = DeterministicInference
    raw_model = ConceptBottleneckALM(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embed=n_embed,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        n_layers=n_layers,
        dropout=dropout,
        out_concepts=dm.n_concepts,
        loss_fn=loss_fn,
        inference=inference_type,
    )

    # 5. Wrap in PyTorch Lightning Module
    pl_model = LightningLM(
        model=raw_model,
        lr=6e-4,
        min_lr=6e-5,
        warmup_steps=100,
        max_steps=1000,
        weight_decay=0.1,
    )

    # 6. Instantiate Trainer
    trainer = pl.Trainer(
        max_steps=500,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_checkpointing=False,
        callbacks=[TQDMProgressBar(refresh_rate=1)],
    )

    # 7. Start Training
    print("--- Launching PyTorch Lightning Trainer ---")
    trainer.fit(model=pl_model, datamodule=dm)
    print("\nTrainer execution finished successfully!")


if __name__ == "__main__":
    main()
