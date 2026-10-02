"""Computes per-concept "lifted" tokens: vocabulary tokens most associated with a concept.

Two selectable metrics (compute_lifted_tokens' `metric` param):
  "lift" (Section 4.4's original metric): lift(w, c) = P(w|c) / P(w), token frequency within a
      concept's documents vs. corpus-wide frequency. Simple, but sample-size-blind: a token with
      a handful of coincidental occurrences can outrank one with thousands of consistent ones,
      since it's a raw ratio with no notion of how much evidence supports it.
  "log_likelihood" (default): Dunning's log-likelihood ratio / G-test (Dunning 1993), the
      standard "keyness" statistic in corpus linguistics for exactly this word-vs-reference-
      corpus over-representation question. Scales with the amount of data behind the estimate,
      so it doesn't confuse a small-sample fluke for a robust association the way raw lift does.

Used by babysteerling.steering as the token-level attribution signal: a token in a document
already tagged with concept c counts as "attributed" to c if it's one of c's lifted tokens.
Built entirely from data the pipeline already produces, no new LLM-tagging stage needed.
"""

import json
import math
import os
from collections import Counter

import torch


def _log_likelihood_ratio(count, concept_total, corpus_count, total_tokens):
    """G-test / Dunning's log-likelihood ratio for token `count` occurrences within a concept's
    `concept_total` tokens, against `corpus_count` occurrences in `total_tokens` corpus-wide: the
    2x2 contingency table of (token vs. not-token) x (this concept vs. rest of corpus), summed
    observed*log(observed/expected) over all four cells. Magnitude only -- doesn't carry the
    sign of the association (see compute_lifted_tokens' lift>1 gate)."""
    a = count
    b = corpus_count - count
    c = concept_total - count
    d = (total_tokens - concept_total) - b
    n = a + b + c + d

    def term(o, e):
        return o * math.log(o / e) if o > 0 else 0.0

    e_a = (a + b) * (a + c) / n
    e_b = (a + b) * (b + d) / n
    e_c = (c + d) * (a + c) / n
    e_d = (c + d) * (b + d) / n
    return 2 * (term(a, e_a) + term(b, e_b) + term(c, e_c) + term(d, e_d))


def compute_lifted_tokens(
    tokens_path,
    doc_records_path,
    output_path=None,
    top_k=50,
    min_support=5,
    metric="log_likelihood",
    direction="positive",
):
    """Ranks vocabulary tokens per concept by `metric` (see module docstring): "lift" (Section
    4.4's original P(w|c)/P(w) ratio) or "log_likelihood" (default -- Dunning's log-likelihood
    ratio, robust to the small-sample noise raw lift is prone to).

    direction="positive" (default): tokens over-represented in the concept's documents (lift > 1)
    -- "lifted" tokens proper. direction="negative": tokens under-represented instead (lift < 1,
    ranked by ascending lift for metric="lift", or by log-likelihood magnitude for
    metric="log_likelihood" -- same small-sample-noise problem applies in reverse here: a token
    with only a handful of total occurrences can look strongly avoided by lift alone just by
    chance, the same way a token with a handful of coincidental occurrences can look strongly
    lifted; "log_likelihood" fixes both directions the same way, by scaling with how much data
    supports the estimate. Call this twice (positive and negative, to separate output_paths) to
    get both.

    tokens_path/doc_records_path: the same steerling_tokens.pt/steerling_concepts.pt written by
    tokenize_dataset(), reused directly.

    Returns {concept_id: [token_id, ...]}, top_k tokens per concept. Each token must appear at
    least min_support times within that concept's documents (a floor against the cheapest cases
    of spurious noise; "log_likelihood" already discounts small samples on its own, but keeps
    this too, for consistency with "lift").

    Idempotent: if output_path exists, loads and returns it instead of recomputing.
    """
    if metric not in ("lift", "log_likelihood"):
        raise ValueError(
            f"unknown metric {metric!r}, expected 'lift' or 'log_likelihood'"
        )
    if direction not in ("positive", "negative"):
        raise ValueError(
            f"unknown direction {direction!r}, expected 'positive' or 'negative'"
        )

    if output_path and os.path.exists(output_path):
        print(f"Found existing {output_path}, skipping lifted-token computation.")
        with open(output_path, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}

    tokens = torch.load(tokens_path)
    doc_records = torch.load(doc_records_path)

    corpus_counts = Counter()  # token_id -> count across the whole corpus
    concept_counts = (
        {}
    )  # concept_id -> Counter(token_id -> count within that concept's documents)
    total_tokens = len(tokens)

    for doc in doc_records:
        doc_tokens = tokens[
            doc["start"] : doc["end"]
        ].tolist()  # this document's own token ids
        corpus_counts.update(doc_tokens)
        for concept_id in doc["concept_ids"]:
            concept_counts.setdefault(concept_id, Counter()).update(doc_tokens)

    lifted_tokens = {}
    for concept_id, counts in concept_counts.items():
        concept_total = sum(counts.values())
        scored = []
        for token_id, count in counts.items():
            if count < min_support:
                continue
            p_token_given_concept = count / concept_total
            p_token = corpus_counts[token_id] / total_tokens
            lift = p_token_given_concept / p_token
            if direction == "positive" and lift <= 1:
                continue
            if direction == "negative" and lift >= 1:
                continue
            if metric == "lift":
                # descending sort of `score` should surface the strongest association first in
                # both directions: for "positive" that's the highest lift; for "negative" it's
                # the lowest (closest to 0), so negate to keep the same sort direction
                score = lift if direction == "positive" else -lift
            else:
                # magnitude only, no sign -- direction is already enforced by the lift gate above
                score = _log_likelihood_ratio(
                    count, concept_total, corpus_counts[token_id], total_tokens
                )
            scored.append((score, token_id))
        scored.sort(reverse=True)  # highest score first
        lifted_tokens[concept_id] = [token_id for _, token_id in scored[:top_k]]

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {str(k): v for k, v in lifted_tokens.items()}, f, indent=2
            )  # JSON keys must be strings
        print(f"Wrote lifted tokens for {len(lifted_tokens)} concepts to {output_path}")

    return lifted_tokens
