"""Atlas-inspired concept dataset pipeline, 4 idempotent stages:

    tag_chunks -> build_concepts -> assign_concepts -> tokenize_dataset

Plus two standalone steps that only need earlier output, not the whole chain:
compute_lifted_tokens (per-concept token stats from the finished dataset, used by
babysteerling.steering) and build_concept_prototypes (synthetic positive/negative/unrelated
text probes per concept, from concepts.json alone).

Implements Guide Labs' Atlas pipeline (see NOTICE) at laptop scale: tag a text corpus with an
LLM, cluster the tags into a concept library, assign concepts back to the corpus, then tokenize.
"""

from .assign_concepts import assign_concepts
from .build_concepts import build_concepts
from .concept_prototypes import build_concept_prototypes
from .lifted_words import compute_lifted_tokens
from .tag_chunks import tag_chunks
from .tokenize_dataset import tokenize_dataset

__all__ = [
    "tag_chunks",
    "build_concepts",
    "assign_concepts",
    "tokenize_dataset",
    "compute_lifted_tokens",
    "build_concept_prototypes",
]
