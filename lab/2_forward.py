from torch_concepts.nn import DeterministicInference

from babysteerling.loader import ConceptDataModule
from babysteerling.loss import TokenLoss, CompositeLoss, ConceptLoss
from babysteerling.metric import TokenAccuracy, ConceptAUC
from babysteerling.model.alm_cb import ConceptBottleneckALM


def main():
    # 1. Initialize and set up the DataModule
    dm = ConceptDataModule(
        data_dir="./data",
        block_size=1024,
        batch_size=32,
        num_workers=4,
    )
    dm.setup("fit")

    # 2. Fetch a single batch from the train_dataloader
    batch = next(iter(dm.train_dataloader()))

    # 3. Inspect keys and tensor shapes
    print("--- Batch Keys ---")
    print(batch.keys())

    print("\n--- Tensor Shapes & Metadata ---")
    print(f"input_ids:    {batch['input_ids'].shape}   # [B, T]")
    print(f"targets:      {batch['targets'].shape}   # [B, T]")
    print(f"known_labels: {batch['known_labels'].shape} # [B, T, n_concepts]")
    print(f"doc_spans:    {len(batch['doc_spans'])} spans across batch")

    # 4. Inspect a sample doc_span tuple (batch_idx, tok_start, tok_end, concept_ids)
    if batch["doc_spans"]:
        print(f"First span:   {batch['doc_spans'][0]}")

    # 5. Instantiate the Model & Loss Function
    # Model hyperparams
    vocab_size = 2000
    block_size = dm.hparams.block_size
    n_embed = 128
    num_heads = 4
    num_kv_heads = 2
    n_layers = 4
    dropout = 0.2
    out_concepts = dm.n_concepts

    # 6. Instantiate CompositeLoss
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

    # 7. Instantiate Model
    inference_type = DeterministicInference
    model = ConceptBottleneckALM(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embed=n_embed,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        n_layers=n_layers,
        dropout=dropout,
        out_concepts=out_concepts,
        loss_fn=loss_fn,
        inference=inference_type,
    )

    # 8. Execute 1 Step via model.step(batch)
    model.eval()
    output = model.step(batch, **{"step": 0})

    # 9. Inspect Results
    print("\n--- Model Output Inspection ---")
    print(f"Total Loss:            {output.loss.item():.4f}")

    print("\n--- Metrics Readout ---")
    eval_metrics = model.loss_fn.compute_metrics()
    for k, v in eval_metrics.items():
        print(f"  {k:<20}: {v:.4f}")

    # 10. Backward Pass Verification
    output.loss.backward()
    print("\nBackward pass completed successfully!")


if __name__ == "__main__":
    main()
