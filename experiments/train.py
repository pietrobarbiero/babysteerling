"""Minimal Hydra + W&B training entry point.

Kept thin: all the real logic (data loading, model architecture, losses, training-loop
utilities) lives in the installed `babysteerling` package. This file just unpacks Hydra's `cfg`
into calls against that package: sample a batch, forward, compute loss, step, occasionally
evaluate and log.

Supports two backbones (`model.backbone_type`): "causal" (default, next-token prediction) and
"diffusion" (masked-diffusion, block-causal attention). The two closures below
(`train_step`/`eval_losses`), plus the final generation call, are the only places that branch
on which one is in use.

Run a single experiment:      python train.py
Override any hyperparameter:  python train.py training.lr=1e-3 model=deep_head
Use the diffusion backbone:   python train.py model=steerling_diffusion
Run a parallel sweep:         python train.py -m model=steerling_gpt,deep_head training.lr=1e-3,3e-4
(see README.md for the full explanation of config overrides and multirun)
"""
import hashlib
import json
import os

# must be set before `import torch`; no-op on CUDA/CPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from babysteerling import diffusion
from babysteerling.data.utils import (
    build_supervision, filter_concepts_by_lifted_tokens, get_batch, load_concept_prototype_tokens,
    load_dataset, load_lifted_token_prototypes, load_lifted_tokens, load_tokenizer,
)
from babysteerling.model import build_model
from babysteerling.training import estimate_diffusion_loss, estimate_loss, get_lr, run_batch, run_diffusion_batch


def log_concept_activation_table(model, sample_ids, tok, concept_names, block_size, device):
    """known_encoder_type='linear_selector' only: logs a W&B table of every (token, activated
    concept) pair for the just-generated sample, so you can inspect which concepts fired where.

    Re-runs the generated ids through the model (chunked to block_size, since the position
    embedding table doesn't cover longer sequences) to recover intermediates['alpha_k'], the per-token
    concept activation tensor. Runs in eval() mode, so the table reflects the model's actual
    learned routing, not one noisy training-time sample.

    A concept counts as "activated" for a token if its k entry is nonzero: that's exactly the
    set LinearSelector routed to (see nn.prototype.PrototypePredictor). One row per (token,
    activated concept), sorted by |activation| within each token, so W&B's table UI can
    sort/filter on concept_name or activation directly.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        k_chunks = []
        for start in range(0, len(sample_ids), block_size):
            chunk = torch.tensor([sample_ids[start:start + block_size]], device=device)
            _, intermediates = model(chunk)
            k_chunks.append(intermediates['alpha_k'][0])  # shape: [chunk_len, n_concepts]
        k_all = torch.cat(k_chunks, dim=0)  # shape: [total_len, n_concepts]
    model.train(was_training)

    columns = ["position", "token_id", "token", "concept_id", "concept_name", "activation"]
    rows = []
    for pos, token_id in enumerate(sample_ids):
        token_str = tok.decode([token_id])
        activated = [(c, k_all[pos, c].item()) for c in k_all[pos].nonzero(as_tuple=True)[0].tolist()]
        activated.sort(key=lambda pair: abs(pair[1]), reverse=True)  # strongest activation first, within this token
        for concept_id, activation in activated:
            rows.append([pos, token_id, token_str, concept_id, concept_names.get(concept_id, f"concept_{concept_id}"), activation])

    wandb.log({"concept_activations": wandb.Table(columns=columns, data=rows)})


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'mps' and cfg.model.known_encoder_type == "product_key":
        # PrototypePredictor's MPS crash fix needs a small Kt (see its forward()); product_key's
        # Kt (100k-1M) is too big for that, so it stays on CPU.
        print("known_encoder_type='product_key': forcing device='cpu' (MPS crash fix needs small Kt)")
        device = 'cpu'
    is_diffusion = cfg.model.backbone_type == "diffusion"

    # resumable checkpoint, keyed by a hash of the resolved config (everything except `wandb`,
    # which is just logging metadata -- not part of what makes two runs "the same experiment")
    # so a crash + rerun of the same command picks back up instead of restarting from step 0
    # (see experiments/train_resilient.sh). Checked before wandb.init() so a resumed run can
    # reuse its previous wandb id (below) instead of starting a new run that shows up as a short,
    # disconnected fragment on the dashboard.
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_hash = hashlib.sha256(
        json.dumps({k: v for k, v in config_dict.items() if k != 'wandb'}, sort_keys=True).encode()
    ).hexdigest()[:16]
    os.makedirs("./checkpoints", exist_ok=True)
    resume_path = os.path.join("./checkpoints", f"resume_{config_hash}.pt")
    ckpt = torch.load(resume_path, map_location=device) if os.path.exists(resume_path) else None

    # W&B logs the fully-resolved config, so every hyperparameter (CLI overrides included) shows
    # up in the run's config panel
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        name=cfg.wandb.name or cfg.model.wandb_name,
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=config_dict,
        id=ckpt.get('wandb_run_id') if ckpt else None,
        resume="must" if ckpt and 'wandb_run_id' in ckpt else None,
    )

    tok, vocab_size, decode = load_tokenizer(cfg.data.data_dir)
    mask_token_id = None
    if is_diffusion:
        # diffusion needs a [MASK] token; add it before computing vocab_size, so the embedding
        # table and head are sized to include it
        mask_token_id = diffusion.ensure_mask_token(tok)
        vocab_size = tok.get_vocab_size()

    tokens, doc_records, n_concepts = load_dataset(cfg.data.data_dir)
    # drop concepts with too little lifted-token signal to be worth a prototype or a model slot;
    # concepts.json/concept_prototypes.json/lifted_tokens.json on disk stay untouched -- this
    # filters fresh, in memory, every time data is loaded to train or evaluate
    doc_records, n_concepts, concepts = filter_concepts_by_lifted_tokens(
        cfg.data.data_dir, doc_records, n_concepts, min_lifted_tokens=cfg.data.min_lifted_tokens,
    )
    n_train = int(0.9 * len(tokens))  # 90/10 train/val split
    doc_starts = [d['start'] for d in doc_records]  # sorted (documents are laid out in order), used for binary search
    print(f"Loaded {len(tokens)} tokens, {len(doc_records)} documents, {n_concepts} known concepts")

    # every encoder except "dense" routes through each concept's own prototype texts
    # (babysteerling.data.babyatlas.build_concept_prototypes) or, if model.predictor_type ==
    # "lifted_tokens", each concept's own top lifted tokens instead; proto_token_ids stays None
    # for dense
    proto_token_ids = None
    if cfg.model.known_encoder_type != "dense":
        orig_concept_ids = [c['orig_concept_id'] for c in concepts]
        if cfg.model.predictor_type == "lifted_tokens":
            proto_token_ids = load_lifted_token_prototypes(
                cfg.data.data_dir, orig_concept_ids, top_k=cfg.model.lifted_top_k,
            )
            if proto_token_ids is None:
                raise FileNotFoundError(
                    "model.predictor_type='lifted_tokens' requires lifted_tokens.json in "
                    f"{cfg.data.data_dir!r}; rebuild the dataset with build_dataset.py"
                )
        else:
            proto_token_ids = load_concept_prototype_tokens(
                cfg.data.data_dir, tok, orig_concept_ids, max_tokens=cfg.model.max_prototype_tokens,
            )
            if proto_token_ids is None:
                raise FileNotFoundError(
                    f"model.known_encoder_type={cfg.model.known_encoder_type!r} requires "
                    f"concept_prototypes.json in {cfg.data.data_dir!r}; rebuild the dataset with "
                    "build_dataset.py atlas.enable_prototypes=true"
                )

    # build_model takes plain kwargs, no Hydra dependency, so this just unpacks the model
    # config group into it
    model = build_model(
        vocab_size=vocab_size, n_concepts=n_concepts, block_size=cfg.data.block_size,
        n_embed=cfg.model.n_embed, num_heads=cfg.model.num_heads, num_kv_heads=cfg.model.num_kv_heads,
        n_layers=cfg.model.n_layers, dropout=cfg.model.dropout, unknown_ratio=cfg.model.unknown_ratio,
        p_epsilon=cfg.model.p_epsilon, unknown_rank=cfg.model.unknown_rank,
        top_k_known=cfg.model.top_k_known, top_k_unknown=cfg.model.top_k_unknown,
        head_type=cfg.model.head.type, tie_weights=cfg.model.head.tie_weights,
        memory_embedding_dims=cfg.model.memory_embedding_dims,
        memory_size=cfg.model.memory_size,
        warmup_steps=cfg.training.warmup_steps,
        init_temperature=cfg.model.init_temperature, final_temperature=cfg.model.final_temperature,
        head_mlp_hidden=cfg.model.head.mlp_hidden,
        backbone_type=cfg.model.backbone_type, diff_block_len=cfg.model.diff_block_len,
        interpretable=cfg.model.interpretable,
        known_encoder_type=cfg.model.known_encoder_type, proto_token_ids=proto_token_ids,
        topk_axis=cfg.model.topk_axis, chunk_size=cfg.model.chunk_size,
        known_key_dim=cfg.model.known_key_dim, use_checkpoint=cfg.model.use_checkpoint,
        candidates_per_token=cfg.model.candidates_per_token,
        predictor_type=cfg.model.predictor_type, lifted_top_k=cfg.model.lifted_top_k,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{n_params:.3f} M params")
    wandb.run.summary['n_params_M'] = n_params

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    start_step = 0
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step'] + 1
        print(f"Resuming from {resume_path} at step {start_step}")

    # steering (Section 6.2) and steering-training (Section 10.2.4) are opt-in: only import
    # babysteerling.steering when cfg.steering.enabled, so a default run never pays for it
    steering_module = None
    lifted_tokens = {}
    if cfg.steering.enabled:
        from babysteerling import steering as steering_module
        lifted_tokens = load_lifted_tokens(cfg.data.data_dir, "positive")
        if not lifted_tokens:
            print("Warning: steering.enabled=true but no lifted_tokens.json found in "
                  f"{cfg.data.data_dir!r}; rebuild the dataset with build_dataset.py to compute "
                  "it. Steering-training steps will fall back to ordinary LM steps until then.")

    def train_step(split, use_steering=False):
        if use_steering:
            return steering_module.run_steering_batch(
                model, tokens, doc_records, doc_starts, n_train, n_concepts, split,
                cfg.data.block_size, cfg.data.batch_size, device, lifted_tokens,
                lambda_respond=cfg.steering.lambda_respond, lambda_express=cfg.steering.lambda_express,
                inj_layer=cfg.steering.inj_layer, tau=cfg.steering.tau, is_diffusion=is_diffusion,
                mask_token_id=mask_token_id, diff_block_len=cfg.model.diff_block_len,
            )
        if is_diffusion:
            return run_diffusion_batch(
                model, tokens, doc_records, doc_starts, n_train, n_concepts, split,
                cfg.data.block_size, cfg.data.batch_size, device, mask_token_id, cfg.model.diff_block_len,
                lambda_concept=cfg.training.lambda_concept, lambda_rec=cfg.training.lambda_rec,
                lambda_indep=cfg.training.lambda_indep, use_concept_loss=cfg.model.use_concept_loss,
            )
        return run_batch(
            model, tokens, doc_records, doc_starts, n_train, n_concepts, split,
            cfg.data.block_size, cfg.data.batch_size, device,
            lambda_concept=cfg.training.lambda_concept, lambda_rec=cfg.training.lambda_rec,
            lambda_indep=cfg.training.lambda_indep, use_concept_loss=cfg.model.use_concept_loss,
        )

    def steering_metrics_fn(split):
        # hook estimate_loss/estimate_diffusion_loss call once per split to attach steering's
        # respond/output-change metrics; None disables it, whether steering is off or a batch
        # has no eligible concept
        if steering_module is None or not lifted_tokens:
            return None
        xb, _, starts = get_batch(tokens, split, cfg.data.block_size, cfg.data.batch_size, n_train, device)
        doc_spans, _ = build_supervision(doc_records, doc_starts, starts, cfg.data.block_size, n_concepts, device)
        return steering_module.steering_ability_metrics(
            model, xb, doc_spans, lifted_tokens, inj_layer=cfg.steering.inj_layer, tau=cfg.steering.tau,
        )

    def eval_losses():
        fn = steering_metrics_fn if steering_module is not None else None
        if is_diffusion:
            return estimate_diffusion_loss(
                model, tokens, doc_records, doc_starts, n_train, n_concepts,
                cfg.data.block_size, cfg.data.batch_size, device, cfg.training.eval_iters,
                mask_token_id, cfg.model.diff_block_len,
                lambda_concept=cfg.training.lambda_concept, lambda_rec=cfg.training.lambda_rec,
                lambda_indep=cfg.training.lambda_indep, steering_metrics_fn=fn,
                use_concept_loss=cfg.model.use_concept_loss,
            )
        return estimate_loss(
            model, tokens, doc_records, doc_starts, n_train, n_concepts,
            cfg.data.block_size, cfg.data.batch_size, device, cfg.training.eval_iters,
            lambda_concept=cfg.training.lambda_concept, lambda_rec=cfg.training.lambda_rec,
            lambda_indep=cfg.training.lambda_indep, steering_metrics_fn=fn,
            use_concept_loss=cfg.model.use_concept_loss,
        )

    for step in range(start_step, cfg.training.max_steps + 1):
        current_lr = get_lr(step, cfg.training.lr, cfg.training.min_lr,
                             cfg.training.warmup_steps, cfg.training.max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        do_steering_step = (
            steering_module is not None and lifted_tokens
            and step > 0 and step % cfg.steering.every_n_steps == 0
        )
        train_loss, _ = train_step('train', use_steering=do_steering_step)

        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()

        if step > 0 and step % cfg.training.checkpoint_interval == 0:
            torch.save({
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                'wandb_run_id': wandb.run.id,
            }, resume_path)

        if step % cfg.training.eval_interval == 0:
            losses = eval_losses()
            print(f"step {step}, train loss {losses['train']['total']:.4f}, "
                  f"val loss {losses['val']['total']:.4f}, "
                  f"train lm_acc {losses['train']['lm_accuracy']:.4f}, "
                  f"val lm_acc {losses['val']['lm_accuracy']:.4f}, "
                  f"train concept_acc_or {losses['train']['concept_accuracy_or']:.4f}, "
                  f"val concept_acc_or {losses['val']['concept_accuracy_or']:.4f}, "
                  f"train concept_acc_tok {losses['train']['concept_accuracy_per_token']:.4f}, "
                  f"val concept_acc_tok {losses['val']['concept_accuracy_per_token']:.4f}, "
                  f"lr {current_lr:.6f}")
            # prefix keys so W&B groups train/*.total and val/*.total as separate lines on the
            # same chart, and each loss component gets its own comparable panel across runs
            wandb.log({
                'step': step,
                'lr': current_lr,
                **{f'train/{k}': v for k, v in losses['train'].items()},
                **{f'val/{k}': v for k, v in losses['val'].items()},
            })

    if os.path.exists(resume_path):
        os.remove(resume_path)  # finished cleanly: don't let a later run resume from this

    # each run gets its own checkpoint, named after the W&B run: a sweep produces many runs,
    # not one to resume. Bundled with the config/vocab_size/n_concepts (not just the state_dict),
    # so it's loadable standalone later for other tests, metrics, or generation without needing
    # to know the hyperparameters build_model was called with.
    run_name = wandb.run.name or f"run_{int(time.time())}"
    os.makedirs("./checkpoints", exist_ok=True)
    ckpt_path = os.path.join("./checkpoints", f"{run_name}.pt")
    torch.save({
        'model': model.state_dict(),
        'config': OmegaConf.to_container(cfg, resolve=True),
        'vocab_size': vocab_size,
        'n_concepts': n_concepts,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # sample a short generation and log it to W&B as text, so qualitative output is visible
    # next to the loss curves
    if is_diffusion:
        sample_ids = model.generate(
            mask_token_id, seq_len=cfg.data.block_size,
            gen_steps=cfg.training.gen_steps, temperature=cfg.training.gen_temperature,
            top_k=cfg.training.gen_top_k,
        )
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        sample_ids = model.generate(
            idx, max_new_tokens=cfg.training.gen_max_new_tokens,
            temperature=cfg.training.gen_temperature, top_k=cfg.training.gen_top_k,
        )[0].tolist()
    sample_text = decode(sample_ids)
    print(sample_text)
    wandb.log({'sample': wandb.Html(f"<pre>{sample_text}</pre>")})

    if cfg.model.known_encoder_type == "linear_selector":
        # concepts (filtered/renumbered by filter_concepts_by_lifted_tokens above) already has
        # the same concept_id numbering the model's activations use -- not concepts.json's raw
        # ids, which wouldn't line up if any concepts got dropped
        concept_names = {c['concept_id']: c['label'] for c in concepts}
        log_concept_activation_table(model, sample_ids, tok, concept_names, cfg.data.block_size, device)

    wandb.finish()


if __name__ == "__main__":
    main()
