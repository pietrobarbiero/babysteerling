"""Stage 0: fetch a raw training corpus and train a tokenizer on it.

Kept separate from the Atlas stages (tag/cluster/assign/tokenize): this is one-time corpus
setup, not concept annotation, and stays the same no matter which annotation hyperparameters a
build_dataset run uses.
"""

import os
import urllib.request

from tokenizers import ByteLevelBPETokenizer


def download_corpus(url, output_path):
    """Downloads a plain-text corpus from `url` to `output_path`.

    Works with any corpus; which one to use is a config concern (see
    experiments/configs/corpus/), not something hardcoded here.

    Idempotent: skips if `output_path` already exists.
    """
    if os.path.exists(output_path):
        print(f"Found existing {output_path}, skipping download.")
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Downloading corpus from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded corpus to {output_path}")


def train_tokenizer(
    input_paths, output_path, vocab_size=2000, boundary_token="<|endoftext|>"
):
    """Trains a byte-level BPE tokenizer on one or more corpus files.

    `input_paths` can be a single path or a list. Pass every source's file when building a
    union dataset (see build_dataset.py), so one shared vocabulary covers all of them.

    `boundary_token` is added as a special token so it survives tokenization intact. It's
    separate from each source's own `document_delimiter` (which only splits that source's raw
    file into documents): boundary_token is the marker this pipeline inserts after every chunk
    in the tokenized stream, so it just needs to be one consistent token tokenize_dataset() can
    look up.

    Idempotent: skips if `output_path` already exists.
    """
    if os.path.exists(output_path):
        print(f"Found existing {output_path}, skipping tokenizer training.")
        return
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    print(
        f"Training a byte-level BPE tokenizer (vocab_size={vocab_size}) on {len(input_paths)} file(s)..."
    )
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=input_paths,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[boundary_token],
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tokenizer.save(output_path)
    print(
        f"Trained tokenizer with vocab_size={tokenizer.get_vocab_size()}, saved to {output_path}"
    )
