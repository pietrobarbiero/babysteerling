# Contributing

Thanks for your interest in babysteerling. This is a small, didactic project. The bar for a
contribution is "does it stay simple and readable", not "does it cover every edge case."

## Development setup

```bash
git clone https://github.com/pyc-team/babysteerling.git
cd babysteerling
pip install -e ".[atlas]"   # editable install; the atlas extra pulls in the dataset-building deps
pre-commit install  # for standardized code formatting when making commits
```

## Repository layout

- `babysteerling/`: the installable package. Layers (`nn/`), models (`model/`), 
  the Atlas dataset pipeline (`data/`), and other utilities.
- `lab/`: the Hydra + Weights & Biases setup for experiments: configs, examples, `run_auto.py`,
  `build_dataset.py`. See `lab/README.md`.
- `0_gpt_chars/`, `1_gpt_tokens/`, `2_atlas/`, `3_steerling/`: standalone, single-file scripts that
  document the build-up to `babysteerling`. They are self-contained on purpose and don't depend on
  the package. Please don't refactor them to import from `babysteerling` (that would break the
  point of reading them top to bottom in isolation), and please don't make `babysteerling`/
  `lab` depend on them either. `build_dataset.py` fetches and builds everything it needs
  on its own (see `lab/README.md`'s "Building a dataset" section).

## Making a change

1. Open an issue or a small PR describing what you're changing and why.
2. Keep changes minimal and consistent with the existing style: flat, commented, explicit shapes
   on tensor operations (see any file in `babysteerling/` for the convention), no abstraction
   beyond what the change actually needs.
3. Sanity check your change runs end to end before opening a PR (there's no CI yet). For example,
   run `lab/run_auto.py` with `training.max_steps` overridden to a small number.
4. Submit a PR against `main`.

By contributing, you agree your contributions are licensed under the project's Apache License,
Version 2.0 (see `LICENSE`).
