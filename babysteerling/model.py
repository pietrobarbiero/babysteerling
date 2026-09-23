"""Model architecture: transformer backbone + Steerling concept bottleneck + head.

Independently implements the architecture from Guide Labs' "Scaling Inherently Interpretable
Language Models" (see NOTICE), at a scale that trains in minutes on a laptop.

The concept bottleneck splits the backbone's hidden state into three parts that add back up to
it, before the final projection to vocabulary logits:

    h_bar = k (known concepts) + u (unknown concepts) + epsilon (residual)

Idea: instead of letting the model use its hidden state however it wants, force part of it
through a small set of human-labeled "known" concepts (see babysteerling.data.atlas for how the
concept library is built), part through a larger set of "unknown" concepts the model discovers on
its own, and let a residual catch whatever's left. Since the final head is linear, every output
logit is an exact sum of a known-concept part, an unknown-concept part, and a residual part. That
is what lets us trace a prediction back to specific concepts.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from .nn.backbone import TransformerModel
from .nn.bottleneck import ConceptBottleneck
from .nn.predictor import LinearEmbeddingToConcept, ReluEmbeddingToConcepts, LinearMemoryPredictor


class BaseLM(nn.Module):
    """Backbone + optional concept bottleneck + head, as one module.

    One nn.Module instead of three separate objects, so model.parameters() and
    model.state_dict() automatically dedupe the tied embedding/head weight.

    interpretable=False skips the bottleneck entirely and forward() returns an empty intermediates
    dict, which compute_losses reads as "score cross-entropy only". Same training loop either way,
    so a no-bottleneck run is a like-for-like baseline for what the bottleneck costs.

    Subclasses set the attention mask and generate(): GPT (causal) and Diffusion (block-causal).
    """

    def __init__(self, vocab_size, block_size, n_embed, num_heads, num_kv_heads, n_layers,
                 dropout, interpretable=True, n_concepts=None, unknown_ratio=3, p_epsilon=0.1,
                 unknown_rank=None, top_k_known=None, top_k_unknown=None, head_type="linear",
                 tie_weights=True, head_mlp_hidden=None,
                 memory_embedding_dims=None, memory_size=None, warmup_steps=200,
                 init_temperature=1.0, final_temperature=0.1,
                 known_encoder_type="dense", proto_token_ids=None, topk_axis=5, chunk_size=4096,
                 known_key_dim=None, use_checkpoint=True, candidates_per_token=25,
                 predictor_type="prototype", lifted_top_k=5):
        super().__init__()
        self.block_size = block_size
        self.backbone = TransformerModel(vocab_size, n_embed, block_size, num_heads, n_layers, dropout, num_kv_heads)
        if interpretable:
            assert n_concepts is not None, "n_concepts is required when interpretable=True"
            self.bottleneck = ConceptBottleneck(
                n_embed, n_concepts, unknown_ratio=unknown_ratio, p_epsilon=p_epsilon,
                unknown_rank=unknown_rank, top_k_known=top_k_known, top_k_unknown=top_k_unknown,
                known_encoder_type=known_encoder_type, proto_token_ids=proto_token_ids,
                backbone=self.backbone, topk_axis=topk_axis,
                chunk_size=chunk_size, key_dim=known_key_dim, use_checkpoint=use_checkpoint,
                candidates_per_token=candidates_per_token,
                predictor_type=predictor_type, lifted_top_k=lifted_top_k,
            )
        else:
            self.bottleneck = None
        tied_embedding = self.backbone.token_embedding_table.weight if tie_weights else None
        if head_type == "linear":
            self.head = LinearEmbeddingToConcept(
                n_embed, vocab_size, tie_weights=tie_weights, tied_embedding=tied_embedding,
            )
        elif head_type == "mlp":
            self.head = ReluEmbeddingToConcepts(
                n_embed, vocab_size, mlp_hidden=head_mlp_hidden,
            )
        elif head_type == "memory":
            self.head = LinearMemoryPredictor(
                n_embed, vocab_size,
                memory_embedding_dims=memory_embedding_dims,
                memory_size=memory_size,
                warmup_steps=warmup_steps,
                init_temperature=init_temperature,
                final_temperature=final_temperature,
            )

        attn_mask = None
        # non-persistent: it's cheap to rebuild and shouldn't be saved into/loaded from checkpoints
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, idx, known_labels=None):
        h = self.backbone(idx, attn_mask=self.attn_mask)  # shape: [B, T, n_embed]
        if self.bottleneck is None:
            return self.head(h), {}  # known_labels accepted and ignored
        h_bar, intermediates = self.bottleneck(h, known_labels=known_labels)
        logits = self.head(h_bar)  # shape: [B, T, vocab_size]
        return logits, intermediates


class GPT(BaseLM):
    """Next-token prediction: causal attention (attn_mask=None), autoregressive sampling."""

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive sampling, one token at a time, cropping context to block_size."""
        was_training = self.training
        self.eval()  # sampling must not apply dropout; MPS also refuses a nonzero dropout_p
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # shape: [B, <=block_size], keep only the last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # shape: [B, T, vocab] -> [B, vocab], last-position logits only

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # shape: [B, top_k], top-k logit values
                logits[logits < v[:, [-1]]] = float('-inf')  # mask out everything below the k-th largest logit

            probs = F.softmax(logits, dim=-1)  # shape: [B, vocab]
            idx_next = torch.multinomial(probs, num_samples=1)  # shape: [B, 1]
            idx = torch.cat((idx, idx_next), dim=1)  # shape: [B, T] -> [B, T+1]
        if was_training:
            self.train()
        return idx


class Diffusion(BaseLM):
    """Masked block diffusion: block-causal attention (bidirectional inside a block of
    `diff_block_len` tokens, causal across blocks), iterative denoising. This class only builds the
    attention mask; corruption and the noise schedule live in babysteerling.diffusion.
    """

    def __init__(self, *args, diff_block_len=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert diff_block_len is not None, "diff_block_len is required for the diffusion backbone"
        self.diff_block_len = diff_block_len
        from .diffusion import build_block_causal_mask  # local import: diffusion.py doesn't need to import nn.py
        attn_mask = build_block_causal_mask(self.block_size, diff_block_len)

        # non-persistent: it's cheap to rebuild and shouldn't be saved into/loaded from checkpoints
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    @torch.no_grad()
    def generate(self, mask_token_id, seq_len, gen_steps=None, temperature=0.8, top_k=50):
        """Block-by-block confidence-based MDM sampler, committing one token per step.

        gen_steps=None (default) commits exactly one position per step, so the step count is
        implicit: diff_block_len steps per block, seq_len forward passes in total. Set it to an int
        to spread that many steps over the sequence instead and commit several positions per step,
        which is faster but see (3) below.

        Four things it has to get right, each of which visibly wrecks the output otherwise:

        1. Decode one block at a time, finishing block b before starting b+1. Attention is
           block-causal, so committing tokens out of block order asks the model to predict block 0
           from nothing.
        2. Commit the *most confident* masked position, not a random one, since committing a
           low-confidence guess early conditions everything after it on that guess.
        3. Commit ONE token per step. Everything committed in the same step comes from a single
           forward pass, so those positions cannot condition on each other, which is what produces
           "said said said" repetition. This is what gen_steps trades away.
        4. Never emit `[MASK]`. It's in the vocabulary for the corruption process, so it's in the
           softmax unless excluded, and since decode() drops special tokens, sampling it silently
           deletes words from the output.

        No KV cache, so every step re-runs the full forward pass.
        """
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()  # sampling must not apply dropout; MPS also refuses a nonzero dropout_p
        x = torch.full((1, seq_len), mask_token_id, dtype=torch.long,
                       device=device)  # shape: [1, seq_len], start fully masked

        num_blocks = max(1, seq_len // self.diff_block_len)
        steps_per_block = None if gen_steps is None else max(1, gen_steps // num_blocks)

        for block in range(num_blocks):
            lo = block * self.diff_block_len
            hi = min(lo + self.diff_block_len, seq_len)

            step = 0
            while True:
                still_masked = (x[0, lo:hi] == mask_token_id).nonzero(as_tuple=True)[0]  # block-local indices
                if len(still_masked) == 0:
                    break  # (1) this block is done; move to the next

                logits, _ = self(x)  # shape: [1, seq_len, vocab_size]
                block_logits = logits[0, lo:hi] / temperature  # shape: [block_len, vocab_size]
                block_logits[:, mask_token_id] = float('-inf')  # (4) never emit the mask token

                probs = F.softmax(block_logits, dim=-1)
                if top_k is not None:
                    v, _ = torch.topk(block_logits, min(top_k, block_logits.size(-1)), dim=-1)
                    probs = torch.where(block_logits < v[:, [-1]], torch.zeros_like(probs), probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize after truncation

                # (2) confidence = the probability mass on each position's own best token
                confidence = probs[still_masked].max(dim=-1).values  # shape: [n_still_masked]
                n_commit = 1 if steps_per_block is None else (  # (3) one position unless gen_steps says otherwise
                    -(-len(still_masked) // max(1, steps_per_block - step))  # ceil, to finish the block on schedule
                )
                pos = still_masked[confidence.topk(n_commit).indices]  # the most confident positions
                x[0, lo + pos] = torch.multinomial(probs[pos], num_samples=1).squeeze(-1)
                step += 1

        if was_training:
            self.train()
        return x[0].tolist()


def build_model(vocab_size, n_concepts, block_size, n_embed=128, num_heads=4, num_kv_heads=2,
                 n_layers=4, dropout=0.2, unknown_ratio=3, p_epsilon=0.1, unknown_rank=None,
                 top_k_known=None, top_k_unknown=None, head_type="linear", tie_weights=True,
                 memory_embedding_dims=None, memory_size=None, warmup_steps=200,
                 init_temperature=1.0, final_temperature=0.1,
                 head_mlp_hidden=None, backbone_type="causal", diff_block_len=None, interpretable=True,
                 known_encoder_type="dense", proto_token_ids=None, topk_axis=5, chunk_size=4096,
                 known_key_dim=None, use_checkpoint=True, candidates_per_token=25,
                 predictor_type="prototype", lifted_top_k=5):
    """Builds a GPT (causal) or Diffusion (block-causal) model from plain keyword arguments.

    interpretable=False drops the concept bottleneck, giving a plain-LM baseline for measuring what
    the bottleneck costs. No config object needed, so it works the same whether the caller uses
    Hydra or not (see experiments/train.py for the adapter that unpacks `cfg.model` into this call).

    known_encoder_type: "dense" (default, a small MLP over the known concept library),
    "linear_selector" or "product_key" (score a subset of concepts against their prototype
    texts), or "prototype_attention" (score all concepts against their prototypes, chunked). See
    babysteerling.nn.prototype. The last three need proto_token_ids (see
    babysteerling.data.utils.load_concept_prototype_tokens).

    predictor_type (linear_selector/product_key only): "prototype" (default) or "lifted_tokens"
    (score against each concept's own top lifted tokens instead -- proto_token_ids must then come
    from babysteerling.data.utils.load_lifted_token_prototypes). See nn.bottleneck's
    _build_known_encoder.
    """
    kwargs = dict(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embed=n_embed,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        n_layers=n_layers,
        dropout=dropout,
        interpretable=interpretable,
        n_concepts=n_concepts,
        unknown_ratio=unknown_ratio,
        p_epsilon=p_epsilon,
        unknown_rank=unknown_rank,
        top_k_known=top_k_known,
        top_k_unknown=top_k_unknown,
        head_type=head_type,
        tie_weights=tie_weights,
        memory_embedding_dims=memory_embedding_dims,
        memory_size=memory_size,
        warmup_steps=warmup_steps,
        init_temperature=init_temperature,
        final_temperature=final_temperature,
        head_mlp_hidden=head_mlp_hidden,
        known_encoder_type=known_encoder_type,
        proto_token_ids=proto_token_ids,
        topk_axis=topk_axis,
        chunk_size=chunk_size,
        known_key_dim=known_key_dim,
        use_checkpoint=use_checkpoint,
        candidates_per_token=candidates_per_token,
        predictor_type=predictor_type,
        lifted_top_k=lifted_top_k,
    )
    if backbone_type.lower() == "causal":
        return GPT(**kwargs)
    elif backbone_type.lower() == "diffusion":
        return Diffusion(**kwargs, diff_block_len=diff_block_len)
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}. Supported types are 'causal' and 'diffusion'.")
