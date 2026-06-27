"""Multi-head causal self-attention — the core primitive of a GPT-2 block.

This module is intentionally standalone: it imports nothing from the block,
FFN, embedding, or full-model code, so it can be unit-tested in isolation.

The computation it performs, for a single attention head, is

        Attention(Q, K, V) = softmax( (Q Kᵀ) / √d_k ) V

with a *causal* mask that forbids any position from attending to positions
that come after it (the defining property of an autoregressive decoder).

Pure PyTorch — no Hugging Face, no external attention kernels.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Configurable multi-head causal self-attention.

    Args:
        n_embd:      embedding / model dimension (the channel size C).
        n_head:      number of attention heads. ``n_embd`` must be divisible
                     by ``n_head`` because the channels are split evenly into
                     heads, each of size ``head_dim = n_embd // n_head``.
        bias:        whether the Q/K/V and output projections use a bias term.
        dropout:     dropout probability applied to the attention weights and
                     to the residual (output) projection.
        max_seq_len: longest sequence length supported; sizes the precomputed
                     causal mask buffer.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        bias: bool = True,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        # The head split is an exact reshape, so the channels must divide evenly.
        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
            )

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head  # == d_k, the per-head dimension

        # A single linear layer produces Q, K and V in one matmul; the output
        # (3 * n_embd) is split into the three projections afterwards. This is
        # exactly equivalent to three separate Linear(n_embd, n_embd) layers,
        # just fused for efficiency.
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # Output projection that mixes the concatenated heads back together.
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: a lower-triangular matrix of ones. Entry (i, j) is 1
        # when query position i is allowed to attend to key position j, i.e.
        # when j <= i. Stored as a buffer (moves with .to(device), not trained).
        # Shape (1, 1, T, T) broadcasts over the batch and head dimensions.
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer(
            "causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """Apply causal self-attention.

        Args:
            x:           input tensor of shape (batch, seq, n_embd).
            return_attn: if True, also return the attention weight tensor of
                         shape (batch, n_head, seq, seq) for inspection/testing.

        Returns:
            Tensor of shape (batch, seq, n_embd); or a (output, weights) tuple
            when ``return_attn`` is True.
        """
        B, T, C = x.shape

        # 1) Project the input to queries, keys and values in one matmul, then
        #    split the last dimension (3 * n_embd) into three (n_embd) chunks.
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)  # each (B, T, C)

        # 2) Split each of Q/K/V into heads and move the head axis next to the
        #    batch axis: (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim).
        #    Each head now attends independently over the sequence.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3) Scaled dot-product scores: Q Kᵀ gives, for every query position,
        #    a similarity to every key position. We divide by √d_k so that the
        #    dot products don't grow with head_dim and push softmax into
        #    vanishingly-small-gradient saturation. Shape: (B, n_head, T, T).
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 4) Causal mask: wherever the mask is 0 (a future key, j > i), set the
        #    score to -inf so that exp(-inf) = 0 in the softmax. This guarantees
        #    a position can never attend to tokens that come after it.
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        # 5) Softmax over the key axis turns scores into a probability
        #    distribution; masked (future) entries become exactly zero.
        att = F.softmax(att, dim=-1)
        weights = att  # keep the pre-dropout weights for inspection
        att = self.attn_dropout(att)

        # 6) Weighted sum of the value vectors, then reassemble the heads back
        #    into a single (B, T, C) tensor and apply the output projection.
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concat heads
        y = self.resid_dropout(self.proj(y))

        if return_attn:
            return y, weights
        return y
