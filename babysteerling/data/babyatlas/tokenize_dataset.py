"""Stage 4: tokenize concept-annotated chunks and align concept IDs to document token spans.

Produces the two artifacts babysteerling.data.utils.load_dataset() expects: a single
concatenated token tensor, and a per-document record of where each document's tokens live in
that tensor plus its ground-truth concept ids.
"""

import json
import os

import torch
from tokenizers import Tokenizer


def tokenize_dataset(
    chunk_concepts_path,
    tokenizer_path,
    tokens_output_path,
    concepts_output_path,
    boundary_token="<|endoftext|>",
):
    """Tokenizes each concept-annotated chunk (from assign_concepts()) and records its token
    span in the concatenated stream, so per-document concept labels can be looked up later
    without re-parsing text.

    `boundary_token` must be the same token prepare.train_tokenizer() was given: it's appended
    after every chunk to mark its boundary, independent of whatever `document_delimiter` each
    source's raw text used.

    Idempotent: skips if both output paths already exist.
    """
    if os.path.exists(tokens_output_path) and os.path.exists(concepts_output_path):
        print(
            f"Found existing {tokens_output_path} and {concepts_output_path}, skipping tokenization."
        )
        return

    print(f"Loading tokenizer from {tokenizer_path}...")
    tok = Tokenizer.from_file(tokenizer_path)
    eot_id = tok.token_to_id(boundary_token)

    print(f"Tokenizing chunks from {chunk_concepts_path}...")
    all_ids = []
    doc_records = []
    with open(chunk_concepts_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            # if one chunk == one document (as it does for a TinyStories-shaped corpus via this
            # pipeline), the boundary token alone marks the boundary; no separate [EOC] token
            # is needed
            ids = tok.encode(chunk["text"]).ids
            ids.append(eot_id)

            # record this document's span in the concatenated token stream, so concept_ids
            # (chunk-level labels) can be looked up per document without re-parsing text
            start = len(all_ids)
            all_ids.extend(ids)
            end = len(all_ids)

            doc_records.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "start": start,
                    "end": end,
                    "concept_ids": chunk["concept_ids"],
                }
            )

    tokens = torch.tensor(all_ids, dtype=torch.long)
    os.makedirs(os.path.dirname(tokens_output_path) or ".", exist_ok=True)
    torch.save(tokens, tokens_output_path)
    torch.save(doc_records, concepts_output_path)

    print(f"Tokenized {len(doc_records)} documents into {len(tokens)} tokens.")
    print(f"Wrote token tensor to {tokens_output_path}")
    print(f"Wrote concept alignment to {concepts_output_path}")
