# Stage 4: tokenize concept-annotated chunks and align concept IDs to document token spans.
import json
import os

import torch
from tokenizers import Tokenizer

chunk_concepts_path = "./data/chunk_concepts.jsonl"
tokenizer_path = "./data/tinystories_tokenizer.json"
tokens_path = "./data/steerling_tokens.pt"
concepts_path = "./data/steerling_concepts.pt"

if os.path.exists(tokens_path) and os.path.exists(concepts_path):
    print(f"Found existing {tokens_path} and {concepts_path}, skipping tokenization.")
else:
    print(f"Loading tokenizer from {tokenizer_path}...")
    tok = Tokenizer.from_file(tokenizer_path)
    eot_id = tok.token_to_id("<|endoftext|>")

    print(f"Tokenizing chunks from {chunk_concepts_path}...")
    all_ids = []
    doc_records = []
    with open(chunk_concepts_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            # each chunk is one full story == one document, so <|endoftext|> alone marks
            # the boundary; no separate [EOC] token needed (unlike the paper's multi-chunk docs)
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
    torch.save(tokens, tokens_path)
    torch.save(doc_records, concepts_path)

    print(f"Tokenized {len(doc_records)} documents into {len(tokens)} tokens.")
    print(f"Wrote token tensor to {tokens_path}")
    print(f"Wrote concept alignment to {concepts_path}")
