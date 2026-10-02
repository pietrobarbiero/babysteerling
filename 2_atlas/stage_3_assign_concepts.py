# Stage 3: assign each chunk's raw tags to final concept IDs (no trained classifier needed
# at this scale); optionally extend coverage to untagged stories via nearest-centroid lookup.
import json
import os
import random
import string

import numpy as np

# Params
ENABLE_SCALE_UP = (
    False  # set True to extend concept coverage to more (untagged) stories
)
NUM_SCALEUP_STORIES = 20000
SCALEUP_SEED = 7
SIMILARITY_FLOOR = 0.3
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

input_path = "./data/tinystories/input.txt"
tags_path = "./data/tags.jsonl"
concepts_path = "./data/concepts.json"
chunk_concepts_path = "./data/chunk_concepts.jsonl"


def normalize_tag(tag):
    tag = tag.lower().replace("-", " ").replace("_", " ")
    tag = tag.translate(str.maketrans("", "", string.punctuation.replace(" ", "")))
    return " ".join(tag.split())


def load_stories(path, n, seed):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    stories = [s.strip() for s in text.split("<|endoftext|>")]
    stories = [s for s in stories if s]
    random.Random(seed).shuffle(stories)
    return stories[:n]


if os.path.exists(chunk_concepts_path):
    print(f"Found existing {chunk_concepts_path}, skipping assignment.")
else:
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
    with open(chunk_concepts_path, "w", encoding="utf-8") as out_f:
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

    if ENABLE_SCALE_UP:
        import torch
        from sentence_transformers import SentenceTransformer

        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        print(
            f"Scale-up enabled: embedding stories with {EMBED_MODEL_NAME} on {device}..."
        )
        embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)

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
        candidates = load_stories(input_path, NUM_SCALEUP_STORIES, SCALEUP_SEED)
        candidates = [s for s in candidates if s not in already_tagged]
        print(f"Embedding {len(candidates)} additional untagged stories...")

        # since everything is unit-normalized, the dot product is cosine similarity;
        # assign each story to its single nearest concept centroid
        story_embeddings = embedder.encode(
            candidates, show_progress_bar=True, normalize_embeddings=True
        )
        similarities = story_embeddings @ concept_centroids.T
        best_concept = similarities.argmax(axis=1)
        best_score = similarities.max(axis=1)

        num_assigned = 0
        with open(chunk_concepts_path, "a", encoding="utf-8") as out_f:
            for story, concept_idx, score in zip(candidates, best_concept, best_score):
                if score < SIMILARITY_FLOOR:
                    continue
                out_f.write(
                    json.dumps(
                        {
                            "chunk_id": next_chunk_id,
                            "text": story,
                            "concept_ids": [concepts[concept_idx]["concept_id"]],
                            "source": "centroid_nn",
                        }
                    )
                    + "\n"
                )
                next_chunk_id += 1
                num_assigned += 1

        print(
            f"Scale-up assigned concepts to {num_assigned}/{len(candidates)} additional stories "
            f"(similarity >= {SIMILARITY_FLOOR})."
        )

    print(f"Wrote chunk-concept assignments to {chunk_concepts_path}")
