# Baby Steerling

A didactic package to develop Interpretable Language Models (ILMs).
It allows to train ILMs on a laptop in a few minutes per step.

The numbered folders are a step by step hands-on guide meant to be read in order.
`babysteerling/` is the main package to develop prototypes, 
while `lab/` provides a configurable environment for experiments.

- **`0_gpt_chars/`**: A bigram model and a tiny GPT, trained character by character on
  TinyShakespeare. No tokenizer, no interpretability, just next character prediction.
- **`1_gpt_tokens/`**: Same GPT, but with a BPE tokenizer, trained on TinyStories. Includes a
  couple of faster training script variants.
- **`2_atlas/`**: Before a model can use concepts, we need to define them. This folder builds a
  small concept labeled version of TinyStories: an LLM tags text chunks, we cluster the tags into
  a concept library, and assign concepts back to the data. Based on Guide Labs' Atlas pipeline
  (see NOTICE), scaled down to run on a laptop.
- **`3_steerling/`**: Single file implementations of the Steerling architecture: a concept
  bottleneck that splits the model's hidden state into known concepts, unknown concepts, and a
  residual, trained with extra losses so predictions can be traced back to concepts.
  `steerling.py` follows the reference architecture (Causal Diffusion backbone, teacher forcing).
  `gpt_steerling.py` uses a plain GPT backbone instead, simpler and faster to iterate on.
  `gpt_steerling_deephead.py` and `gpt_steerling_no_supervision.py` are small ablations (a deeper
  head; concept losses turned off).
- **`babysteerling/`**: Package to develop prototypes of interpretable language models. See NOTICE for attribution.
- **`lab/`**: A Hydra + Weights & Biases environment to run experiments. Use it to train
  models, build datasets, log results, and compare runs, including parallel sweeps. See
  `experiments/README.md` for the full how-to.

## Setup

```bash
mamba create -n babysteerling python
pip install -r requirements.txt      # deps for the standalone numbered folders
pip install -e ".[atlas]"            # editable install of babysteerling, needed for experiments/
```

## Running a single experiment

```bash
cd experiments
python build_dataset_auto.py   # first time only: downloads TinyStories and builds a concept dataset
python run_auto.py  # train a default model with default hyperparameters, logs to W&B
```

## License

Apache License 2.0 (see `LICENSE`). See `NOTICE` for attribution to Guide Labs' Steerling and
Atlas work, which this project implements independently at a much smaller scale.
