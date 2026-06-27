"""Tests for the standalone causal self-attention module.

These exercise the math (scaled dot-product + causal masking), the
configurable head splitting, the shape contract, and the isolation
requirement (no block/model imports).
"""

import ast
import inspect

import pytest
import torch

from gpt2.attention import CausalSelfAttention


def test_forward_preserves_shape_for_multiple_n_head():
    """forward maps (B, T, C) -> (B, T, C) for several head counts."""
    batch, seq, n_embd = 3, 5, 8
    x = torch.randn(batch, seq, n_embd)
    # 8 is divisible by each of these head counts.
    for n_head in (1, 2, 4):
        attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head)
        y = attn(x)
        assert y.shape == (batch, seq, n_embd), (
            f"n_head={n_head} produced {tuple(y.shape)}"
        )


def test_requires_n_embd_divisible_by_n_head():
    """n_embd must be an exact multiple of n_head."""
    # 8 % 3 != 0 -> rejected.
    with pytest.raises(ValueError):
        CausalSelfAttention(n_embd=8, n_head=3)
    # 8 % 4 == 0 -> accepted (no raise).
    CausalSelfAttention(n_embd=8, n_head=4)


def test_causal_mask_zeros_future_attention_weights():
    """No position may attend to a later position: future weights are exactly 0,
    and the surviving (causal) weights still form a valid distribution."""
    batch, seq, n_embd, n_head = 2, 6, 8, 2
    attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head, dropout=0.0)
    attn.eval()
    x = torch.randn(batch, seq, n_embd)
    _, weights = attn(x, return_attn=True)  # (B, n_head, T, T)

    assert weights.shape == (batch, n_head, seq, seq)

    # Strict upper triangle = future positions. Every such weight must be 0.
    future = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
    future_weights = weights[:, :, future]
    assert torch.all(future_weights == 0), "future positions received nonzero attention"

    # Sanity: rows still sum to 1 (softmax over the allowed positions),
    # so the implementation is not trivially zeroing everything.
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    # The diagonal (a position attending to itself) is always allowed -> > 0.
    diag = torch.diagonal(weights, dim1=-2, dim2=-1)
    assert torch.all(diag > 0)


def test_output_at_position_is_independent_of_future_inputs():
    """End-to-end causality: changing a token only affects outputs at or
    after its position, never earlier ones."""
    torch.manual_seed(0)
    seq, n_embd, n_head = 5, 8, 4
    attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head, dropout=0.0)
    attn.eval()

    x = torch.randn(1, seq, n_embd)
    y1 = attn(x)

    # Perturb only the LAST token.
    x2 = x.clone()
    x2[:, -1, :] += 10.0
    y2 = attn(x2)

    # Earlier positions cannot see the last token -> outputs unchanged.
    assert torch.allclose(y1[:, :-1, :], y2[:, :-1, :], atol=1e-6)
    # The last position itself does change (it attends to the perturbed token).
    assert not torch.allclose(y1[:, -1, :], y2[:, -1, :], atol=1e-6)


def test_module_does_not_import_block_or_model():
    """The attention module is self-contained: it must not import the
    block/model layers it is a dependency of."""
    source = inspect.getsource(
        __import__("gpt2.attention", fromlist=["attention"])
    )
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.append(node.module)
            imported.extend(alias.name for alias in node.names)

    banned = ("block", "model")
    offenders = [
        name for name in imported if any(b in name.lower() for b in banned)
    ]
    assert not offenders, f"attention module must not import: {offenders}"
