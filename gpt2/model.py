"""Assembled GPT-2 decoder stack — the full autoregressive language model.

This wires the standalone :class:`~gpt2.attention.CausalSelfAttention` primitive
into a complete GPT-2-style decoder:

* a :class:`TransformerBlock` — pre-LayerNorm → attention → residual, then
  pre-LayerNorm → position-wise feed-forward → residual;
* a :class:`GPT2` model — token + learned positional embeddings, a stack of
  ``n_layer`` blocks, a final LayerNorm, and a language-model head that
  projects back to vocabulary logits.

Everything is parameterized by :class:`GPT2Config`, so a tiny educational model
fits on a laptop. Pure PyTorch — no Hugging Face. Attention is *imported* from
the predecessor module and never reimplemented here.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from gpt2.attention import CausalSelfAttention


@dataclass
class GPT2Config:
    """Hyperparameters that fully describe the model's shape.

    Defaults are the canonical GPT-2 (124M) values; tests and small experiments
    override them to build a laptop-sized model.
    """

    vocab_size: int = 50257  # number of distinct tokens the model can emit
    n_layer: int = 12        # number of stacked transformer blocks
    n_head: int = 12         # attention heads per block (must divide n_embd)
    n_embd: int = 768        # embedding / model channel dimension
    block_size: int = 1024   # maximum sequence length (context window)


class TransformerBlock(nn.Module):
    """One GPT-2 decoder block: two residual sub-layers, each pre-normalized.

    The "pre-LayerNorm" placement (normalize *before* the sub-layer, add the
    raw input back) is what keeps deep stacks numerically stable — the residual
    path is an unobstructed identity highway, so gradients flow cleanly.
    """

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        # --- attention sub-layer ---
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        # Attention is the imported primitive; sized strictly from the config so
        # head-count and channel width are never hardcoded.
        self.attn = CausalSelfAttention(
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            max_seq_len=cfg.block_size,
        )
        # --- feed-forward sub-layer ---
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        # Minimal position-wise MLP: expand ~4x, GELU non-linearity, project back.
        # The final Linear is mlp[-1]; zeroing it removes this sub-layer's
        # contribution, exposing the residual identity (see tests).
        hidden = 4 * cfg.n_embd
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x + sublayer(norm(x)) for each sub-layer — the residual wiring.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """The assembled GPT-2 decoder: embeddings → blocks → final norm → lm_head."""

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg

        # Token embedding: maps each token id to an n_embd vector.
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # Learned positional embedding: one vector per absolute position.
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)

        # The decoder stack — exactly n_layer blocks.
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layer))

        # Final LayerNorm applied to the last block's output before the head.
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # Language-model head: projects the n_embd representation to vocab logits.
        # No bias, because its weight is *tied* to the token embedding below.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying (a defining GPT-2 property): the LM head reuses the token
        # embedding matrix — input and output token representations share one set
        # of parameters. They are the SAME tensor, not two copies.
        self.lm_head.weight = self.wte.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Map (batch, seq) token ids to (batch, seq, vocab_size) logits."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.cfg.block_size}"
            )

        # Positions 0..T-1; positional embedding broadcasts over the batch.
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)  # (B, T, n_embd)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)            # final LayerNorm before projecting to logits
        logits = self.lm_head(x)    # (B, T, vocab_size)
        return logits
