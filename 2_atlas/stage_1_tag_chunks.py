# Stage 1: sample TinyStories entries and LLM-tag each with free-form concept tags.
import json
import os
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(1337)

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Params
NUM_STORIES = 5000
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BATCH_SIZE = 16
MAX_NEW_TOKENS = 150

input_path = "./data/tinystories/input.txt"
tags_path = "./data/tags.jsonl"

PROMPT_TEMPLATE = (
    "Read the short children's story below and list 5 to 10 free-form tags describing it: "
    "its characters, setting, theme or moral, plot events, and tone. "
    "Respond with ONLY a JSON array of short lowercase tag strings, no other text.\n\n"
    "Story:\n{story}\n\nTags:"
)


def load_stories(path, n):
    # each story is already delimited by <|endoftext|>, and is short enough to treat
    # as a single chunk (no sentence-splitting/concatenation needed, unlike the paper)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    stories = [s.strip() for s in text.split("<|endoftext|>")]
    stories = [s for s in stories if s]
    random.shuffle(stories)
    return stories[:n]


def parse_tags(output_text):
    # pull the JSON array out of the completion; the model may wrap it in extra prose
    match = re.search(r"\[.*\]", output_text, re.DOTALL)
    if not match:
        return None
    try:
        tags = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        return None
    tags = [t.strip().lower() for t in tags if t.strip()]
    return tags or None


if os.path.exists(tags_path):
    print(f"Found existing {tags_path}, skipping tagging.")
else:
    print("Loading stories...")
    stories = load_stories(input_path, NUM_STORIES)
    print(f"Sampled {len(stories)} stories.")

    print(f"Loading {MODEL_NAME} on {device}...")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    # left padding so every sequence in a batch ends at the same position,
    # letting us slice out just the generated continuation below
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.to(device)
    model.eval()

    num_failed = 0
    os.makedirs(os.path.dirname(tags_path), exist_ok=True)
    with open(tags_path, "w", encoding="utf-8") as out_f:
        for batch_start in range(0, len(stories), BATCH_SIZE):
            batch = stories[batch_start : batch_start + BATCH_SIZE]
            # build one chat-formatted prompt per story, batched for throughput
            prompts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": PROMPT_TEMPLATE.format(story=story)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for story in batch
            ]
            inputs = tok(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(device)

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

            for i, (story, completion) in enumerate(zip(batch, completions)):
                tags = parse_tags(completion)
                if tags is None:
                    num_failed += 1
                    continue
                chunk_id = batch_start + i
                out_f.write(
                    json.dumps({"chunk_id": chunk_id, "text": story, "tags": tags})
                    + "\n"
                )

            print(
                f"Tagged {min(batch_start + BATCH_SIZE, len(stories))}/{len(stories)} stories "
                f"({num_failed} failed so far)"
            )

    print(
        f"Done. {num_failed}/{len(stories)} stories failed to parse and were skipped."
    )
    print(f"Wrote tags to {tags_path}")
