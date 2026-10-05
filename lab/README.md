# Baby Steerling Lab

The [Hydra](https://hydra.cc) + [Weights & Biases](https://wandb.ai) configurable environment to run experiments.
`run_auto.py` and `build_dataset_auto.py` just unpack a Hydra config and assemble data, model, and training pipelines.
The other scripts (`1_data.py`, `2_forward.py`, `3_trainer.py`, `4_pyc_style_model.py`) 
are simple, minimal examples to understand each component of `babysteerling`.

## Setup

From the repo root:
```bash
pip install -e ".[atlas]"   # editable install of babysteerling, incl. dataset-building deps
wandb login                 # one-time; skip if you plan to use mode=offline (see below)
```

## Building a dataset

Before training for the first time you need to build a concept-labeled dataset. 
The default config builds a small version of TinyStories:
```bash
cd lab
python build_dataset_auto.py
```
This downloads a raw corpus, trains a tokenizer on it, then runs the four Atlas stages (tag,
cluster, assign, tokenize) in order, writing everything to `atlas.output_dir` (default `./data`).
Every stage is idempotent: it's skipped if already done, so re-running after a partial failure,
or after changing a later stage's hyperparameter, doesn't redo finished work.

You can override data-generation hyperparameters as follows:
```bash
python build_dataset_auto.py atlas.num_documents=2000 atlas.k=80
```
`run_auto.py`'s default config already points `data.data_dir` at this same `./data`, so once the
build finishes, `python run_auto.py` just works.

**Which corpus gets built and how it's processed are two separate configs.**
`configs/corpus/tinystories.yaml` says what/where the data is: `sources`, a list of
`{name, url, document_delimiter}` entries, plus the shared `tokenizer_vocab_size` and
`boundary_token`. `configs/atlas/default.yaml` says how it's processed: sample size, which models
to use, clustering/dedup thresholds, and the two LLM prompt templates for tagging and labeling concepts.

To build from a different corpus, copy `configs/corpus/tinystories.yaml` to a new file (e.g.
`configs/corpus/my_corpus.yaml`) and replace its `sources` entry. If your corpus isn't children's
stories, also copy `configs/atlas/default.yaml` to `configs/atlas/my_corpus.yaml` and adjust
`tag_prompt_template`/`label_prompt_template`. Then run:
```bash
python build_dataset_auto.py corpus=my_corpus atlas=my_corpus
```

**To train on several corpora at once**, add more entries to `sources`:
```yaml
sources:
  - name: tinystories
    url: https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
    document_delimiter: "<|endoftext|>"
  - name: my_other_corpus
    url: https://example.com/my_other_corpus.txt
    document_delimiter: "\n\n\n"   # can differ per source
```
Each source is downloaded and tagged independently, but all sources' tags are combined into one
shared concept library, and every source is assigned concepts from that same library. This is
what makes it a real union: a concept means the same thing no matter which source a chunk came
from, rather than just concatenating unrelated datasets. `boundary_token` (default
`<|endoftext|>`) is separate from `document_delimiter`: it's the single marker the pipeline
inserts after every chunk once tokenized, and it doesn't need to match any source's delimiter.

## Running a single experiment

To train a default model with default hyperparameters, just run:
```bash
cd lab
python run_auto.py
```
This uses the defaults in `configs/config.yaml`.

Override any hyperparameter from the command line, no file editing needed:
```bash
python run_auto.py training.lr=1e-3 training.max_steps=2000
python run_auto.py model=dlm_steerling
```

Each run saves a checkpoint to `./checkpoints/babysteerling-tests/<run-id>/last.ckpt`, and
logs a short sample generation to W&B at the end.


## Adjusting configs / adding a new variant

Each config group is a folder under `configs/` (e.g., `data/`, `model/`, `training/`). 
To change a value permanently, edit the YAML file directly. To add a new
architecture/loss variant add its own config copying the existing pattern.

## Running experiments in parallel (sweeps)

To run multiple experiments in parallel execute
```bash
bash run_sweep.sh
```
This script calls all models listed in `experiment/test_sweep.yaml`.
To add a new model to the parallel run, add a configuration file under `variant/` 
and add the model to the list in `test_sweep.yaml`.
