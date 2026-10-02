from torch.distributions import Bernoulli, OneHotCategorical
from torch.nn import Identity, Linear
from torch_concepts import EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    ParametricCPD,
    LinearEmbeddingToConcept,
    BayesianNetwork,
    DeterministicInference,
)

from babysteerling.loader import ConceptDataModule
from babysteerling.nn.backbone import TransformerModel, TokensToEmbeddings


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

    tokens_to_embedding = TokensToEmbeddings(vocab_size, n_embed, block_size)
    backbone = TransformerModel(
        n_embed, block_size, num_heads, n_layers, dropout, num_kv_heads
    )

    input_var = EmbeddingVariable("input", distribution=Delta, size=n_embed)
    latent_var = EmbeddingVariable("latent", distribution=Delta, size=n_embed)
    concepts = ConceptVariable(
        "concepts",
        distribution=Bernoulli,
        size=1,
        members=[c["label"] for c in dm.concepts],
    )
    tasks = ConceptVariable(
        "tokens",
        distribution=OneHotCategorical,
        size=1,
        members=[f"token_{i}" for i in range(vocab_size)],
    )

    input_cpd = ParametricCPD(input_var, parametrization=Identity(), parents=[])
    backbone_cpd = ParametricCPD(
        latent_var, parametrization=backbone, parents=[input_var]
    )
    concept_cpd = ParametricCPD(
        concepts,
        parametrization={"logits": LinearEmbeddingToConcept(n_embed, out_concepts)},
        parents=[latent_var],
    )
    task_cpd = ParametricCPD(
        tasks,
        parametrization={"logits": Linear(out_concepts, vocab_size)},
        parents=[concepts],
    )

    model = BayesianNetwork(
        [input_var, latent_var, concepts, tasks],
        [input_cpd, backbone_cpd, concept_cpd, task_cpd],
    )
    inference_engine = DeterministicInference(model)

    x = tokens_to_embedding(batch["input_ids"])
    output = inference_engine.query(query=["concepts", "tokens"], evidence={"input": x})
    print("--- Inference Results ---")
    print(f"{output.logits['concepts'].shape}   # [B, T, n_concepts]")
    print(f"{output.logits['tokens'].shape}   # [B, T, vocab_size]")


if __name__ == "__main__":
    main()
