# Stage 2: cluster raw tags into a canonical, human-labeled, deduplicated concept library.
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

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Params
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LABEL_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
K = 150
MIN_CLUSTER_SIZE = 3
TAGS_PER_LABEL_PROMPT = 15
DEDUP_THRESHOLD = 0.9
BATCH_SIZE = 16
MAX_NEW_TOKENS = 100

tags_path = "./data/tags.jsonl"
concepts_path = "./data/concepts.json"

LABEL_PROMPT_TEMPLATE = (
    "The following words/phrases were assigned to children's stories that share a common "
    "concept:\n{tags}\n\n"
    'Respond with ONLY a JSON object with two fields: "label" (a concise 1-6 word name for '
    'this shared concept) and "description" (a one-sentence description). No other text.'
)


def normalize_tag(tag):
    # collapse formatting variants (hyphens/underscores/punctuation) so identical
    # tags written differently don't get split into separate clusters
    tag = tag.lower().replace("-", " ").replace("_", " ")
    tag = tag.translate(str.maketrans("", "", string.punctuation.replace(" ", "")))
    return " ".join(tag.split())


def parse_json_object(output_text):
    # pull the JSON object out of the completion; the model may wrap it in extra prose
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


class UnionFind:
    # tracks merge groups for near-duplicate concept dedup below
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


if os.path.exists(concepts_path):
    print(f"Found existing {concepts_path}, skipping concept building.")
else:
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

    print(f"Embedding tags with {EMBED_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    tag_embeddings = embedder.encode(
        unique_tags, show_progress_bar=True, normalize_embeddings=True
    )

    print(f"Clustering into K={K} clusters...")
    kmeans = KMeans(n_clusters=K, random_state=1337, n_init="auto")
    cluster_labels = kmeans.fit_predict(tag_embeddings)

    # group tags (and their embeddings, needed later to find each cluster's centroid-nearest tags) by cluster
    clusters = {}
    for tag, tag_emb, cluster_id in zip(unique_tags, tag_embeddings, cluster_labels):
        clusters.setdefault(cluster_id, {"tags": [], "embeddings": []})
        clusters[cluster_id]["tags"].append(tag)
        clusters[cluster_id]["embeddings"].append(tag_emb)

    # drop small clusters as noise instead of the paper's LLM-coherence scoring
    surviving = {
        cid: c for cid, c in clusters.items() if len(c["tags"]) >= MIN_CLUSTER_SIZE
    }
    print(
        f"{len(surviving)}/{len(clusters)} clusters survive the min-size filter "
        f"(>= {MIN_CLUSTER_SIZE} tags)."
    )

    print(f"Loading {LABEL_MODEL_NAME} on {device} for cluster labeling...")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained(LABEL_MODEL_NAME, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(LABEL_MODEL_NAME, torch_dtype=dtype)
    model.to(device)
    model.eval()

    cluster_ids = list(surviving.keys())
    raw_concepts = []
    num_failed = 0
    for batch_start in range(0, len(cluster_ids), BATCH_SIZE):
        batch_cids = cluster_ids[batch_start : batch_start + BATCH_SIZE]
        prompts = []
        for cid in batch_cids:
            # sample the tags closest to the cluster centroid as the most representative evidence
            centroid = kmeans.cluster_centers_[cid]
            tags = surviving[cid]["tags"]
            embs = np.array(surviving[cid]["embeddings"])
            dists = np.linalg.norm(embs - centroid, axis=1)
            nearest = [tags[i] for i in np.argsort(dists)[:TAGS_PER_LABEL_PROMPT]]
            prompt_text = LABEL_PROMPT_TEMPLATE.format(tags=", ".join(nearest))
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
                max_new_tokens=MAX_NEW_TOKENS,
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
            f"Labeled {min(batch_start + BATCH_SIZE, len(cluster_ids))}/{len(cluster_ids)} "
            f"clusters ({num_failed} failed so far)"
        )

    print(
        f"Deduplicating {len(raw_concepts)} labeled concepts (cosine threshold {DEDUP_THRESHOLD})..."
    )
    label_embeddings = embedder.encode(
        [c["label"] for c in raw_concepts], normalize_embeddings=True
    )
    # union any pair of concepts whose labels are near-duplicates in embedding space
    sim_matrix = cosine_similarity(label_embeddings)
    uf = UnionFind(len(raw_concepts))
    for i in range(len(raw_concepts)):
        for j in range(i + 1, len(raw_concepts)):
            if sim_matrix[i, j] >= DEDUP_THRESHOLD:
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

    os.makedirs(os.path.dirname(concepts_path), exist_ok=True)
    with open(concepts_path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, indent=2)
    print(f"Wrote concept library to {concepts_path}")
