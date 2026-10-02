"""Stage 3: assign each chunk's raw tags to final concept IDs (no trained classifier needed at
this scale); optionally extend coverage to untagged documents via nearest-centroid lookup.
"""

import json
import os

import numpy as np

from .build_concepts import normalize_tag
from .tag_chunks import load_documents


def assign_concepts(
    tags_path,
    concepts_path,
    output_path,
    input_path=None,
    document_delimiter="<|endoftext|>",
    enable_scale_up=False,
    num_scaleup_documents=20000,
    scaleup_seed=7,
    similarity_floor=0.3,
    embed_model_name="all-MiniLM-L6-v2",
    device=None,
):
    """Maps each tagged chunk's raw tags to final concept IDs via direct lookup (the tag ->
    concept mapping built by build_concepts()), writing one JSON line per chunk to
    `output_path`.

    If `enable_scale_up=True`, also embeds additional untagged documents from `input_path` and
    assigns each to its nearest concept centroid: a cheap way to extend coverage beyond the
    LLM-tagged sample without more LLM calls, at the cost of noisier, single-concept labels.
    The "source" field marks which path produced each row ("llm_tag" vs "centroid_nn").

    Idempotent: skips if `output_path` already exists.
    """
    if os.path.exists(output_path):
        print(f"Found existing {output_path}, skipping assignment.")
        return

    print(f"Loading {tags_path} and {concepts_path}...")
    chunks = []
    with open(tags_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    with open(concepts_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    # invert concepts.json's member_tags lists into a direct tag -> concept_id lookup
    tag_to_concept = {}
    for concept in concepts:
        for tag in concept["member_tags"]:
            tag_to_concept[tag] = concept["concept_id"]

    next_chunk_id = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for chunk in chunks:
            # a chunk's concepts = union of concepts any of its raw tags map to (OR-aggregation)
            norm_tags = {normalize_tag(t) for t in chunk["tags"]}
            concept_ids = sorted(
                {tag_to_concept[t] for t in norm_tags if t in tag_to_concept}
            )
            out_f.write(
                json.dumps(
                    {
                        "chunk_id": next_chunk_id,
                        "text": chunk["text"],
                        "concept_ids": concept_ids,
                        "source": "llm_tag",
                    }
                )
                + "\n"
            )
            next_chunk_id += 1

    print(f"Assigned concepts to {next_chunk_id} LLM-tagged chunks via tag lookup.")

    if enable_scale_up:
        import torch
        from sentence_transformers import SentenceTransformer

        device = device or (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        print(
            f"Scale-up enabled: embedding documents with {embed_model_name} on {device}..."
        )
        embedder = SentenceTransformer(embed_model_name, device=device)

        # a concept's centroid = mean embedding of its member tags (re-normalized to unit length)
        concept_centroids = np.stack(
            [
                embedder.encode(concept["member_tags"], normalize_embeddings=True).mean(
                    axis=0
                )
                for concept in concepts
            ]
        )
        concept_centroids /= np.linalg.norm(concept_centroids, axis=1, keepdims=True)

        already_tagged = {chunk["text"] for chunk in chunks}
        candidates = load_documents(
            input_path,
            num_scaleup_documents,
            delimiter=document_delimiter,
            seed=scaleup_seed,
        )
        candidates = [d for d in candidates if d not in already_tagged]
        print(f"Embedding {len(candidates)} additional untagged documents...")

        # since everything is unit-normalized, the dot product is cosine similarity;
        # assign each document to its single nearest concept centroid
        doc_embeddings = embedder.encode(
            candidates, show_progress_bar=True, normalize_embeddings=True
        )
        similarities = doc_embeddings @ concept_centroids.T
        best_concept = similarities.argmax(axis=1)
        best_score = similarities.max(axis=1)

        num_assigned = 0
        with open(output_path, "a", encoding="utf-8") as out_f:
            for document, concept_idx, score in zip(
                candidates, best_concept, best_score
            ):
                if score < similarity_floor:
                    continue
                out_f.write(
                    json.dumps(
                        {
                            "chunk_id": next_chunk_id,
                            "text": document,
                            "concept_ids": [concepts[concept_idx]["concept_id"]],
                            "source": "centroid_nn",
                        }
                    )
                    + "\n"
                )
                next_chunk_id += 1
                num_assigned += 1

        print(
            f"Scale-up assigned concepts to {num_assigned}/{len(candidates)} additional documents "
            f"(similarity >= {similarity_floor})."
        )

    print(f"Wrote chunk-concept assignments to {output_path}")
