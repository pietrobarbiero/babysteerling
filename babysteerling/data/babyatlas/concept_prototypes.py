"""Standalone stage: generates synthetic text prototypes per concept, to sanity-check a
concept's separability outside the training corpus.

For each concept (its label and description from build_concepts()), an LLM writes short text
chunks of three types:
  - positive:  clearly expresses the concept.
  - negative:  a similar situation where the concept clearly does NOT happen (e.g. for
               "deceptive", an honest moment), not just any unrelated text.
  - unrelated: a different topic entirely.

Each chunk is scored against the concept's own label+description embedding via cosine
similarity, giving a set of prototypes per concept, tagged with type and similarity.

Not part of the core tag_chunks -> build_concepts -> assign_concepts -> tokenize_dataset chain:
this only needs concepts.json, so it can run any time after build_concepts(), e.g. as a
diagnostic or as evaluation data for babysteerling.steering.
"""

import json
import os
import re

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

PROTOTYPE_TYPES = ("positive", "negative", "unrelated")

DEFAULT_PROMPT_TEMPLATES = {
    "positive": (
        "Write {n} short (1-3 sentence) excerpts, in the style of a children's story, that each "
        'clearly and strongly express the concept "{label}" ({description}). '
        "Respond with ONLY a JSON array of strings, no other text."
    ),
    "negative": (
        "Write {n} short (1-3 sentence) excerpts, in the style of a children's story, each set in "
        'a similar kind of situation to the concept "{label}" ({description}), but where that '
        "concept clearly does NOT happen -- show its opposite or absence, not just an unrelated "
        "topic. Respond with ONLY a JSON array of strings, no other text."
    ),
    "unrelated": (
        "Write {n} short (1-3 sentence) excerpts, in the style of a children's story, each about "
        "an ordinary everyday topic (e.g. baking cookies, a rainy afternoon, counting stars, "
        'cleaning a room) with no connection at all to the concept "{label}" ({description}). '
        "Respond with ONLY a JSON array of strings, no other text."
    ),
}


def parse_text_array(output_text, expected_n):
    """Pulls JSON array(s) of strings out of an LLM completion, truncated to expected_n. The
    model sometimes splits requested items across several separate `[...]` arrays instead of
    one; every bracketed span is parsed independently and pooled, rather than assuming there's
    just one."""
    texts = []
    for match in re.findall(r"\[.*?\]", output_text, re.DOTALL):
        try:
            items = json.loads(match)
        except json.JSONDecodeError:
            continue
        if isinstance(items, list):
            texts.extend(t.strip() for t in items if isinstance(t, str) and t.strip())
    return texts[:expected_n]


def _pad_to_batch_size(items, batch_size):
    """Pads a list (e.g. the last batch or a retry round) up to exactly batch_size items, by
    repeating its last element.

    Some smaller batch sizes have been observed to hard-crash PyTorch's MPS backend during
    generate() (a native Metal crash, not catchable). Full-size batches run reliably, so
    padding every batch to that size sidesteps the crash instead of guessing which shapes are
    safe.

    Returns (padded_items, num_real); only keep the first num_real outputs.
    """
    num_real = len(items)
    if num_real >= batch_size:
        return items, num_real
    return items + [items[-1]] * (batch_size - num_real), num_real


def _per_type_count(n_proto):
    """n_proto // 3, the same count for all three types (any remainder is dropped). Downstream
    consumers treat positive/negative/unrelated as fixed anchors on a -1/0/+1 activation axis,
    so each type needs the exact same count, not just "as even as possible"."""
    per_type = n_proto // 3
    if n_proto % 3 != 0:
        print(
            f"n_proto={n_proto} isn't divisible by 3; using {per_type} per type "
            f"({per_type * 3} total) so every type has an equal, fixed count."
        )
    return per_type


def build_concept_prototypes(
    concepts_path,
    output_path,
    n_proto=15,
    embed_model_name="all-MiniLM-L6-v2",
    generation_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    batch_size=16,
    max_new_tokens=300,
    max_retries=3,
    device=None,
    prompt_templates=None,
):
    """For each concept in concepts_path, generates exactly n_proto // 3 short text chunks for
    each of positive/negative/unrelated, and scores each against the concept's own
    label+description embedding via cosine similarity.

    Writes {concept_id: {"positive": [{"text", "similarity"}, ...], "negative": [...],
    "unrelated": [...]}} to output_path. Every type's list has exactly n_proto // 3 items for
    every concept, a hard requirement. A (concept, type) that comes up short is retried
    (sampling instead of greedy decoding after the first attempt, since retrying greedily would
    just repeat the same failure); if still short after max_retries, it's padded by cycling
    through whatever succeeded (logged). This is what lets a downstream consumer treat the
    three types as fixed-size, fixed-order anchors on a -1/0/+1 axis.

    `prompt_templates`, if given, overrides DEFAULT_PROMPT_TEMPLATES per type (each needs
    {n}/{label}/{description} placeholders), the same way tag_prompt_template/
    label_prompt_template do for the other stages.

    Idempotent: skips if `output_path` already exists.
    """
    if os.path.exists(output_path):
        print(f"Found existing {output_path}, skipping concept prototype generation.")
        return

    templates = {**DEFAULT_PROMPT_TEMPLATES, **(prompt_templates or {})}
    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    # MPS has been observed to hard-crash during generate() for this model, even at the full
    # configured batch size, so generation always runs on CPU when MPS is the detected device.
    # Slower, but this is a one-time data-prep stage, not a hot path. Embedding (a much simpler
    # model) has never shown this issue, so it still uses the fast device.
    gen_device = "cpu" if device == "mps" else device

    with open(concepts_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    print(f"Loaded {len(concepts)} concepts from {concepts_path}.")

    per_type = _per_type_count(n_proto)

    print(
        f"Loading {generation_model_name} on {gen_device}"
        + (
            " (MPS generation is unreliable; see comment above)"
            if device == "mps"
            else ""
        )
        + "..."
    )
    dtype = torch.bfloat16 if gen_device in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained(generation_model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(
        generation_model_name, torch_dtype=dtype
    )
    gen_model.to(gen_device)
    gen_model.eval()

    # collected[(concept_id, ptype)] accumulates successfully-parsed texts across retry rounds;
    # pending tracks (concept, ptype, how_many_more_are_needed) for the current round
    collected = {
        (c["concept_id"], ptype): [] for c in concepts for ptype in PROTOTYPE_TYPES
    }
    pending = [
        (concept, ptype, per_type) for concept in concepts for ptype in PROTOTYPE_TYPES
    ]

    for attempt in range(max_retries + 1):
        if not pending:
            break
        do_sample = (
            attempt > 0
        )  # first pass greedy; retries sample, since retrying greedily
        # would just repeat the same failure
        if attempt > 0:
            print(
                f"Retry {attempt}/{max_retries}: {len(pending)} concept x type slot(s) still short..."
            )

        still_pending = []
        for batch_start in range(0, len(pending), batch_size):
            raw_batch = pending[batch_start : batch_start + batch_size]
            batch, num_real = _pad_to_batch_size(raw_batch, batch_size)
            prompts = [
                tok.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": templates[ptype].format(
                                n=needed,
                                label=concept["label"],
                                description=concept["description"],
                            ),
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for concept, ptype, needed in batch
            ]
            inputs = tok(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(gen_device)

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tok.pad_token_id,
            )
            if do_sample:
                gen_kwargs["temperature"] = 0.8
            with torch.no_grad():
                output_ids = gen_model.generate(**inputs, **gen_kwargs)

            # strip the (left-padded) prompt tokens, keep only the generated continuation
            new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
            completions = tok.batch_decode(new_tokens, skip_special_tokens=True)

            for (concept, ptype, needed), completion in zip(
                batch[:num_real], completions[:num_real]
            ):
                texts = parse_text_array(completion, expected_n=needed)
                collected[(concept["concept_id"], ptype)].extend(texts)
                still_needed = needed - len(texts)
                if still_needed > 0:
                    still_pending.append((concept, ptype, still_needed))

            print(
                f"Generated {min(batch_start + batch_size, len(pending))}/{len(pending)} "
                f"concept x type prompts this round."
            )

        pending = still_pending

    num_padded = 0
    for (concept_id, ptype), texts in collected.items():
        if len(texts) >= per_type:
            continue
        if not texts:
            print(
                f"Warning: concept {concept_id} got zero successful '{ptype}' generations "
                f"after {max_retries} retries; using an empty placeholder."
            )
            texts.append("")
        originals = list(texts)  # snapshot of what actually succeeded, cycled to pad
        while len(texts) < per_type:
            texts.append(originals[len(texts) % len(originals)])
            num_padded += 1
    if num_padded:
        print(
            f"Padded {num_padded} prototype slot(s) that fell short after retries (see warnings above)."
        )

    print(f"Embedding prototypes with {embed_model_name}...")
    embedder = SentenceTransformer(embed_model_name, device=device)

    prototypes = {}
    for concept in concepts:
        concept_id = concept["concept_id"]
        concept_text = f"{concept['label']}: {concept['description']}"
        by_type = {}
        for ptype in PROTOTYPE_TYPES:
            texts = collected[(concept_id, ptype)]
            embeddings = embedder.encode(
                [concept_text] + texts, normalize_embeddings=True
            )
            concept_emb, text_embs = embeddings[0:1], embeddings[1:]
            similarities = cosine_similarity(concept_emb, text_embs)[
                0
            ]  # shape: [per_type]
            by_type[ptype] = [
                {"text": text, "similarity": float(sim)}
                for text, sim in zip(texts, similarities)
            ]
        prototypes[concept_id] = by_type

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {str(k): v for k, v in prototypes.items()}, f, indent=2
        )  # JSON keys must be strings

    total_written = sum(
        len(texts) for by_type in prototypes.values() for texts in by_type.values()
    )
    print(
        f"Wrote {total_written} prototypes ({per_type} per type) across {len(prototypes)} concepts to {output_path}"
    )
