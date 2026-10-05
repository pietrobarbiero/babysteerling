"""Data loading for a concept-annotated corpus built by babysteerling.data.atlas (or anything
that writes the same steerling_tokens.pt / steerling_concepts.pt / concepts.json files).

Plain functions, no torch Dataset/DataLoader: a batch is a set of random token windows (like a
standard nanoGPT loader), plus a lookup for which documents overlap each window. That lookup
doesn't fit a map-style Dataset well, so plain functions are simpler here.
"""

import bisect
import json
import os

import torch
from tokenizers import Tokenizer


def combine_jsonl(input_paths, output_path):
    """Concatenates several JSONL files into one, renumbering each row's `chunk_id` in order.

    Used by build_dataset.py to merge tagged/assigned corpus sources into the single
    tags.jsonl/chunk_concepts.jsonl that build_concepts()/tokenize_dataset() expect, so the
    rest of the pipeline just sees one file.

    Idempotent: skips if `output_path` already exists.
    """
    if os.path.exists(output_path):
        print(f"Found existing {output_path}, skipping combine.")
        return
    next_chunk_id = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for input_path in input_paths:
            with open(input_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    row = json.loads(line)
                    row["chunk_id"] = next_chunk_id
                    out_f.write(json.dumps(row) + "\n")
                    next_chunk_id += 1
    print(
        f"Combined {len(input_paths)} file(s) into {output_path} ({next_chunk_id} rows)."
    )


def load_tokenizer(data_dir, filename="tokenizer.json"):
    """Load the BPE tokenizer used to build the dataset. Returns the tokenizer object, its
    vocab size, and a decode function for turning generated ids back into text."""
    tok = Tokenizer.from_file(os.path.join(data_dir, filename))
    vocab_size = tok.get_vocab_size()
    decode = lambda ids: tok.decode(ids)
    return tok, vocab_size, decode


def load_dataset(data_dir):
    """Loads the token stream, per-document concept labels, and the concept library.

    tokens: 1D LongTensor, the whole corpus concatenated (documents separated by <|endoftext|>).
    doc_records: list of {chunk_id, start, end, concept_ids}, one per document: its [start, end)
        span in `tokens` and its ground-truth concept ids.
    n_concepts: size of the known-concept library, used to size the bottleneck's known head and
        the label tensors build_supervision() builds.
    """
    tokens = torch.load(
        os.path.join(data_dir, "steerling_tokens.pt")
    )  # shape: [N_total_tokens]
    doc_records = torch.load(os.path.join(data_dir, "steerling_concepts.pt"))
    with open(os.path.join(data_dir, "concepts.json")) as f:
        concept_library = json.load(f)
    n_concepts = len(concept_library)
    return tokens, doc_records, n_concepts


def filter_concepts_by_lifted_tokens(
    data_dir, doc_records, n_concepts, min_lifted_tokens=5
):
    """Drops concepts with fewer than min_lifted_tokens lifted tokens (not enough token-level
    signal for Section 4.4's lift metric to mean anything -- not worth a prototype or a slot in
    the model), renumbering the survivors contiguously. Call this explicitly, right after
    load_dataset(), when loading data to train or evaluate a model.

    Purely in-memory: concepts.json, concept_prototypes.json, and lifted_tokens.json on disk are
    never touched, so this recomputes its filtering fresh every call rather than persisting it.

    concept_id is a positional index everywhere downstream (the model's concept embedding table,
    load_concept_prototype_tokens' lookup), so a dropped concept must not leave a gap -- that's
    why survivors get renumbered instead of just leaving holes.

    Returns (doc_records, n_concepts, concepts):
      doc_records: same shape as the input, with each document's concept_ids remapped to the
          filtered numbering (ids for dropped concepts are simply absent).
      n_concepts: size of the filtered concept library.
      concepts: the filtered, renumbered concept list. Each entry keeps its on-disk id under
          'orig_concept_id' (pass those to load_concept_prototype_tokens) while 'concept_id' is
          rewritten to the new, dense id (use that with 'label' for concept_id -> label lookups,
          e.g. the W&B concept-activation table).

    A no-op (concepts returned as-is, just with 'orig_concept_id' added) if lifted_tokens.json
    doesn't exist -- an older dataset, or one built before lifted tokens existed.
    """
    with open(os.path.join(data_dir, "concepts.json")) as f:
        concept_library = json.load(f)
    lifted_tokens = load_lifted_tokens(data_dir, "positive")
    if not lifted_tokens:
        concepts = [{**c, "orig_concept_id": c["concept_id"]} for c in concept_library]
        return doc_records, n_concepts, concepts

    keep = [
        c
        for c in concept_library
        if len(lifted_tokens.get(c["concept_id"], [])) >= min_lifted_tokens
    ]
    if len(keep) < len(concept_library):
        print(
            f"Dropping {len(concept_library) - len(keep)} concept(s) with fewer than "
            f"{min_lifted_tokens} lifted tokens ({len(keep)}/{len(concept_library)} remain)."
        )

    concepts = []
    old_to_new = {}
    for new_id, c in enumerate(keep):
        old_to_new[c["concept_id"]] = new_id
        c = dict(c)
        c["orig_concept_id"] = c["concept_id"]
        c["concept_id"] = new_id
        concepts.append(c)

    doc_records = [
        {
            **doc,
            "concept_ids": [
                old_to_new[c] for c in doc["concept_ids"] if c in old_to_new
            ],
        }
        for doc in doc_records
    ]
    return doc_records, len(concepts), concepts, lifted_tokens


PROTOTYPE_VALUE_ORDER = (
    "negative",
    "unrelated",
    "positive",
)  # matches a fixed -1/0/+1 activation axis


def load_concept_prototype_tokens(
    data_dir, tokenizer, concept_ids, filename="concept_prototypes.json", max_tokens=32
):
    """Loads per-concept prototype texts (babysteerling.data.babyatlas.build_concept_prototypes)
    and tokenizes them with the corpus's own BPE tokenizer.

    concept_ids: each output row's on-disk concept id, in order -- pass
    [c['orig_concept_id'] for c in concepts] (see filter_concepts_by_lifted_tokens) so a filtered
    concept library still looks up the right prototype text per surviving concept.

    Returns an int32 tensor [len(concept_ids), 3, per_type, Tp]. The 3 axis is always
    PROTOTYPE_VALUE_ORDER (negative, unrelated, positive). A concept missing from the file gets
    an all-pad row.

    Tp is capped at max_tokens. This matters more than it looks: nn.prototype's Selectors gather
    per (candidate, token position), so one long, uncapped prototype anywhere in the library
    would multiply into every gather, not just its own row (an uncapped Tp near 300 has blown
    this up to double-digit GB for a modest batch). int32 instead of int64 halves that further.

    Returns None if the file doesn't exist, so callers that don't use a prototype-based encoder
    never have to check for it.
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        by_concept = {int(k): v for k, v in json.load(f).items()}

    per_type = len(next(iter(by_concept.values()))[PROTOTYPE_VALUE_ORDER[0]])

    all_ids = []
    for concept_id in concept_ids:
        by_type = by_concept.get(concept_id, {})
        for ptype in PROTOTYPE_VALUE_ORDER:
            items = by_type.get(ptype) or [{"text": ""}] * per_type
            for item in items:
                all_ids.append(tokenizer.encode(item["text"]).ids[:max_tokens])

    Tp = max((len(ids) for ids in all_ids), default=1) or 1
    padded = torch.tensor(
        [ids + [0] * (Tp - len(ids)) for ids in all_ids], dtype=torch.int32
    )
    return padded.view(len(concept_ids), len(PROTOTYPE_VALUE_ORDER), per_type, Tp)


def load_lifted_tokens(data_dir, direction):
    """Loads the per-concept lifted-token stats from babysteerling.data.atlas.compute_lifted_tokens
    (Section 4.4's lift metric), the token-level concept attribution babysteerling.steering uses.
    Returns {} if the dataset predates lifted tokens or steering was never enabled, so callers
    that don't need steering never have to check.
    """
    if direction not in ("positive", "negative"):
        raise ValueError(
            f"direction must be 'positive' or 'negative', got {direction!r}"
        )

    if direction == "positive":
        path = os.path.join(data_dir, "lifted_tokens.json")
    else:
        path = os.path.join(data_dir, "lifted_tokens_negative.json")

    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


LIFTED_VALUE_ORDER = (
    "negative",
    "positive",
)  # matches a fixed -1/+1 activation axis, no "unrelated"


def load_lifted_token_prototypes(data_dir, concept_ids, top_k=5):
    """Loads each concept's top positive and negative lifted tokens (babysteerling.data.babyatlas.
    compute_lifted_tokens) as single-token prototypes for nn.prototype.LiftedTokenPredictor --
    real corpus tokens instead of concept_prototypes.json's LLM-generated prototype sentences.

    concept_ids: each output row's on-disk concept id, in order -- pass
    [c['orig_concept_id'] for c in concepts] (see filter_concepts_by_lifted_tokens), same
    convention as load_concept_prototype_tokens.

    Returns an int64 tensor [len(concept_ids), 2, top_k]. Axis 1 is LIFTED_VALUE_ORDER
    (negative, positive) -- no "unrelated" middle category, since a lifted token is by
    construction one or the other. A concept with fewer than top_k tokens in a direction (or
    missing from that file entirely) gets its remaining slots padded with -1, NOT 0 -- token id 0
    is '<|endoftext|>' (the document-boundary token), a real, frequent corpus token that
    genuinely does show up as a top lifted token for some concepts (verified: several concepts'
    negative lifted tokens include it), so 0 can't double as a "no token here" sentinel the way
    it safely can for load_concept_prototype_tokens's LLM-generated prose (which realistically
    never contains the literal string "<|endoftext|>"). -1 can never collide with a real token
    id (always >= 0); LiftedTokenPredictor clamps it to a dummy valid id only for the embedding
    lookup, after computing its mask from the unclamped -1.

    Returns None if neither lifted_tokens.json nor lifted_tokens_negative.json exists.
    """
    by_direction = {d: load_lifted_tokens(data_dir, d) for d in LIFTED_VALUE_ORDER}
    if not any(by_direction.values()):
        return None

    rows = []  # [direction][concept] -> [top_k] token ids, padded with -1
    for direction in LIFTED_VALUE_ORDER:
        lifted = by_direction[direction]
        for concept_id in concept_ids:
            ids = lifted.get(concept_id, [])[:top_k]
            rows.append(ids + [-1] * (top_k - len(ids)))

    n = len(concept_ids)
    tensor = torch.tensor(rows, dtype=torch.long).view(
        len(LIFTED_VALUE_ORDER), n, top_k
    )
    return tensor.permute(1, 0, 2).contiguous()  # [n, 2, top_k]


def overlapping_docs(doc_records, doc_starts, window_start, window_end):
    """Finds every document whose token span overlaps [window_start, window_end).

    Documents sit contiguously in `tokens` (one ends where the next starts), so the overlap set
    is always a short run of doc_records. `doc_starts` (sorted start offsets) lets us binary
    search straight to that run instead of scanning the whole corpus.

    Returns (local_start, local_end, concept_ids) tuples, with local_start/local_end relative to
    window_start.
    """
    # first candidate document: the last one starting at or before window_start
    i = max(bisect.bisect_right(doc_starts, window_start) - 1, 0)
    spans = []
    while i < len(doc_records) and doc_records[i]["start"] < window_end:
        d = doc_records[i]
        s, e = max(d["start"], window_start), min(
            d["end"], window_end
        )  # intersect with window
        if e > s:
            spans.append(
                (s - window_start, e - window_start, d["concept_ids"])
            )  # -> window-local offsets
        i += 1
    return spans
