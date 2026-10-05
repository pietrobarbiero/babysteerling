"""Stage 2: cluster raw tags into a canonical, human-labeled, deduplicated concept library.

Turns the noisy, high-recall tag pool from tag_chunks() into something usable: embed each
unique tag, cluster similar ones together, drop clusters too small to be a real concept, have an
LLM name each surviving cluster, then merge clusters whose names turn out to be near-duplicates.
"""

import json
import os
import re
import string

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_LABEL_PROMPT_TEMPLATE = (
    "The following words/phrases were assigned to children's stories that share a common "
    "concept:\n{tags}\n\n"
    'Respond with ONLY a JSON object with two fields: "label" (a concise 1-6 word name for '
    'this shared concept) and "description" (a one-sentence description). No other text.'
)


def normalize_tag(tag):
    """Collapse formatting variants (hyphens/underscores/punctuation) so identical tags written
    differently don't get split into separate clusters."""
    tag = tag.lower().replace("-", " ").replace("_", " ")
    tag = tag.translate(str.maketrans("", "", string.punctuation.replace(" ", "")))
    return " ".join(tag.split())


def parse_json_object(output_text):
    """Pull a JSON object out of a (possibly noisy) LLM completion."""
    match = re.search(r"\{.*\}", output_text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "label" not in obj or "description" not in obj:
        return None
    if not isinstance(obj["label"], str) or not isinstance(obj["description"], str):
        return None
    return obj


class _UnionFind:
    """Tracks merge groups for the near-duplicate concept dedup below."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_concepts(
    tags_path,
    output_path,
    embed_model_name="all-MiniLM-L6-v2",
    label_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    k=150,
    min_cluster_size=3,
    tags_per_label_prompt=15,
    dedup_threshold=0.9,
    batch_size=16,
    max_new_tokens=100,
    seed=1337,
    device=None,
    label_prompt_template=DEFAULT_LABEL_PROMPT_TEMPLATE,
):
    """Clusters the raw tags from tag_chunks() into a canonical, human-labeled, deduplicated
    concept library, written as JSON to `output_path`.

    `label_prompt_template` must contain a `{tags}` placeholder; override it for a corpus that
    doesn't fit the default "children's stories" framing.

    Idempotent: skips if `output_path` already exists.
    """
    if os.path.exists(output_path):
        print(f"Found existing {output_path}, skipping concept building.")
        return

    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    print(f"Loading {tags_path}...")
    chunks = []
    with open(tags_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    # dedupe to unique normalized tags; clustering runs on tag types, not occurrences
    tag_to_chunks = {}
    for chunk in chunks:
        for raw_tag in chunk["tags"]:
            norm_tag = normalize_tag(raw_tag)
            if not norm_tag:
                continue
            tag_to_chunks.setdefault(norm_tag, []).append(chunk["chunk_id"])

    unique_tags = list(tag_to_chunks.keys())
    print(f"{len(unique_tags)} unique normalized tags from {len(chunks)} chunks.")

    print(f"Embedding tags with {embed_model_name}...")
    embedder = SentenceTransformer(embed_model_name, device=device)
    tag_embeddings = embedder.encode(
        unique_tags, show_progress_bar=True, normalize_embeddings=True
    )

    print(f"Clustering into K={k} clusters...")
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    cluster_labels = kmeans.fit_predict(tag_embeddings)

    # group tags (and their embeddings, needed later to find each cluster's centroid-nearest tags) by cluster
    clusters = {}
    for tag, tag_emb, cluster_id in zip(unique_tags, tag_embeddings, cluster_labels):
        clusters.setdefault(cluster_id, {"tags": [], "embeddings": []})
        clusters[cluster_id]["tags"].append(tag)
        clusters[cluster_id]["embeddings"].append(tag_emb)

    # drop small clusters as noise instead of an LLM-coherence scoring pass
    surviving = {
        cid: c for cid, c in clusters.items() if len(c["tags"]) >= min_cluster_size
    }
    print(
        f"{len(surviving)}/{len(clusters)} clusters survive the min-size filter "
        f"(>= {min_cluster_size} tags)."
    )

    print(f"Loading {label_model_name} on {device} for cluster labeling...")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained(label_model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(label_model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    cluster_ids = list(surviving.keys())
    raw_concepts = []
    num_failed = 0
    for batch_start in range(0, len(cluster_ids), batch_size):
        batch_cids = cluster_ids[batch_start : batch_start + batch_size]
        prompts = []
        for cid in batch_cids:
            # sample the tags closest to the cluster centroid as the most representative evidence
            centroid = kmeans.cluster_centers_[cid]
            tags = surviving[cid]["tags"]
            embs = np.array(surviving[cid]["embeddings"])
            dists = np.linalg.norm(embs - centroid, axis=1)
            nearest = [tags[i] for i in np.argsort(dists)[:tags_per_label_prompt]]
            prompt_text = label_prompt_template.format(tags=", ".join(nearest))
            prompts.append(
                tok.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(
            device
        )
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        # strip the (left-padded) prompt tokens, keep only the generated continuation
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        completions = tok.batch_decode(new_tokens, skip_special_tokens=True)

        for cid, completion in zip(batch_cids, completions):
            obj = parse_json_object(completion)
            if obj is None:
                num_failed += 1
                label, description = f"concept_{cid}", ""
            else:
                label, description = obj["label"], obj["description"]
            raw_concepts.append(
                {
                    "cluster_id": int(cid),
                    "label": label,
                    "description": description,
                    "member_tags": surviving[cid]["tags"],
                }
            )

        print(
            f"Labeled {min(batch_start + batch_size, len(cluster_ids))}/{len(cluster_ids)} "
            f"clusters ({num_failed} failed so far)"
        )

    print(
        f"Deduplicating {len(raw_concepts)} labeled concepts (cosine threshold {dedup_threshold})..."
    )
    label_embeddings = embedder.encode(
        [c["label"] for c in raw_concepts], normalize_embeddings=True
    )
    # union any pair of concepts whose labels are near-duplicates in embedding space
    sim_matrix = cosine_similarity(label_embeddings)
    uf = _UnionFind(len(raw_concepts))
    for i in range(len(raw_concepts)):
        for j in range(i + 1, len(raw_concepts)):
            if sim_matrix[i, j] >= dedup_threshold:
                uf.union(i, j)

    groups = {}
    for i in range(len(raw_concepts)):
        groups.setdefault(uf.find(i), []).append(i)

    concepts = []
    for group_indices in groups.values():
        # keep the label/description from the group's largest (best-supported) cluster
        # rather than re-labeling the merge with another LLM call
        largest = max(group_indices, key=lambda i: len(raw_concepts[i]["member_tags"]))
        member_tags = sorted(
            {t for i in group_indices for t in raw_concepts[i]["member_tags"]}
        )
        concepts.append(
            {
                "concept_id": len(concepts),
                "label": raw_concepts[largest]["label"],
                "description": raw_concepts[largest]["description"],
                "member_tags": member_tags,
            }
        )

    print(f"Merged {len(raw_concepts)} clusters into {len(concepts)} concepts.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, indent=2)
    print(f"Wrote concept library to {output_path}")
