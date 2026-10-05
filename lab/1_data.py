from babysteerling.loader import ConceptDataModule


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


if __name__ == "__main__":
    main()
