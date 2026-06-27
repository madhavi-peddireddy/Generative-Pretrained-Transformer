"""Tests for the assembled GPT-2 decoder stack.

These exercise the full forward path end-to-end: token + positional
embeddings, a stack of pre-LayerNorm transformer blocks, a final LayerNorm,
and the language-model head that projects to vocabulary logits. They cover the
shape contract, the residual wiring of a single block, autoregressive
(causal) validity through the complete model, config parameterization, and
the requirement that attention is *imported* from the predecessor module
rather than reimplemented here.
"""

import ast
import inspect

import pytest
import torch

from gpt2.model import GPT2, GPT2Config, TransformerBlock


def _tiny_config(**overrides):
    """A laptop-sized config; overrides let each test vary one dimension."""
    cfg = dict(vocab_size=37, n_layer=2, n_head=4, n_embd=16, block_size=8)
    cfg.update(overrides)
    return GPT2Config(**cfg)


def test_transformer_block_preserves_shape():
    """A block maps (B, T, n_embd) -> (B, T, n_embd)."""
    cfg = _tiny_config()
    block = TransformerBlock(cfg)
    block.eval()
    x = torch.randn(2, cfg.block_size, cfg.n_embd)
    y = block(x)
    assert y.shape == (2, cfg.block_size, cfg.n_embd)


def test_transformer_block_is_residual():
    """Zeroing both sub-layer output projections collapses the block to the
    identity -- which can ONLY happen if each sub-layer is wired as a residual
    (x + sublayer(x)) rather than replacing x. A non-residual block would
    output ~0 here, so this distinguishes the correct wiring from the wrong one.
    """
    cfg = _tiny_config()
    block = TransformerBlock(cfg)
    block.eval()
    x = torch.randn(2, cfg.block_size, cfg.n_embd)
    with torch.no_grad():
        # Attention's output projection -> attention sub-layer contributes 0.
        block.attn.proj.weight.zero_()
        block.attn.proj.bias.zero_()
        # FFN's final linear (last module in the mlp Sequential) -> FFN
        # sub-layer contributes 0.
        block.mlp[-1].weight.zero_()
        block.mlp[-1].bias.zero_()
        y = block(x)
    assert torch.allclose(y, x, atol=1e-6), "block is not residual around its sub-layers"


def test_forward_returns_vocab_logits_shape():
    """End-to-end: (batch, seq) token ids -> (batch, seq, vocab_size) logits."""
    cfg = _tiny_config()
    model = GPT2(cfg)
    model.eval()
    batch, seq = 3, 5
    idx = torch.randint(0, cfg.vocab_size, (batch, seq))
    logits = model(idx)
    assert logits.shape == (batch, seq, cfg.vocab_size)


def test_logits_are_causal_through_full_stack():
    """Autoregressive validity through the ASSEMBLED model: altering tokens
    strictly after position t must not change the logits at positions <= t,
    while the first altered position's logits DO change (sanity that the
    perturbation is real)."""
    torch.manual_seed(0)
    cfg = _tiny_config(block_size=8)
    model = GPT2(cfg)
    model.eval()

    seq = 6
    idx = torch.randint(0, cfg.vocab_size, (1, seq))
    with torch.no_grad():
        logits1 = model(idx)

    t = 2
    idx2 = idx.clone()
    # Change every token strictly after position t (guaranteed different ids).
    idx2[:, t + 1 :] = (idx2[:, t + 1 :] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits2 = model(idx2)

    # Positions 0..t never see the future -> identical logits.
    assert torch.allclose(
        logits1[:, : t + 1, :], logits2[:, : t + 1, :], atol=1e-5
    ), "logits at/before t changed when only later tokens were altered"
    # The first changed position consumed a different token -> its logits move.
    assert not torch.allclose(
        logits1[:, t + 1, :], logits2[:, t + 1, :], atol=1e-5
    )


@pytest.mark.parametrize("n_layer,vocab", [(1, 11), (3, 50)])
def test_model_built_from_parameterizable_config(n_layer, vocab):
    """The model honours its config: block count == n_layer and the logit
    dimension == vocab_size."""
    cfg = _tiny_config(n_layer=n_layer, vocab_size=vocab)
    model = GPT2(cfg)
    assert len(model.blocks) == n_layer
    idx = torch.randint(0, vocab, (2, 4))
    out = model(idx)
    assert out.shape == (2, 4, vocab)


def test_rejects_sequence_longer_than_block_size():
    """Sequence length is bounded by block_size: exactly block_size is allowed,
    one token past it is rejected (both boundaries)."""
    cfg = _tiny_config(block_size=8)
    model = GPT2(cfg)
    model.eval()
    ok = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    model(ok)  # must not raise
    too_long = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    with pytest.raises(ValueError):
        model(too_long)


def test_model_imports_attention_and_does_not_reimplement_it():
    """The model must IMPORT CausalSelfAttention from the predecessor module
    and not redefine it here; and the block's attention must actually be that
    imported class."""
    import gpt2.model as m

    tree = ast.parse(inspect.getsource(m))

    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module
        and "attention" in node.module
        and any(alias.name == "CausalSelfAttention" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert imported, "model must import CausalSelfAttention from the attention module"

    classdefs = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert "CausalSelfAttention" not in classdefs, "model must not reimplement attention"

    # The block genuinely uses the imported attention class.
    from gpt2.attention import CausalSelfAttention

    block = TransformerBlock(_tiny_config())
    assert isinstance(block.attn, CausalSelfAttention)
