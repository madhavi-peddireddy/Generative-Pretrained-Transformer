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


@pytest.mark.parametrize("n_embd,n_head", [(24, 6), (32, 8)])
def test_config_wires_n_embd_and_n_head(n_embd, n_head):
    """The model must read n_embd and n_head FROM THE CONFIG, not hardcode the
    tiny defaults (16 and 4). We build with values that differ from both
    defaults and assert the embedding channel width and the per-block attention
    head-count actually reflect the config — a model that ignored cfg.n_embd /
    cfg.n_head would either keep the wrong internal dims or fail to run here.
    """
    assert (n_embd, n_head) != (16, 4)  # guard: must differ from the defaults
    cfg = _tiny_config(n_embd=n_embd, n_head=n_head)
    model = GPT2(cfg)

    # Token + positional embedding channels must equal cfg.n_embd.
    assert model.wte.weight.shape == (cfg.vocab_size, n_embd)
    assert model.wpe.weight.shape == (cfg.block_size, n_embd)

    # Every block's attention must be constructed from cfg.n_head / cfg.n_embd.
    for block in model.blocks:
        assert block.attn.n_embd == n_embd
        assert block.attn.n_head == n_head
        # head_dim is a derived sanity check that the split used these values.
        assert block.attn.head_dim == n_embd // n_head

    # And the assembled stack still runs end-to-end with the non-default dims.
    idx = torch.randint(0, cfg.vocab_size, (2, 4))
    assert model(idx).shape == (2, 4, cfg.vocab_size)


def test_lm_head_weight_is_tied_to_token_embedding():
    """Weight tying is a defining GPT-2 property: the LM head reuses the token
    embedding matrix. They must be the SAME parameter object — an independently
    initialised lm_head would fail this `is` identity check.
    """
    cfg = _tiny_config()
    model = GPT2(cfg)
    assert model.lm_head.weight is model.wte.weight


def test_final_layernorm_exists_and_is_applied_before_lm_head():
    """A final LayerNorm (ln_f) must exist AND actually run on the forward path
    feeding the LM head. Two independent checks make a model that omits ln_f
    fail:

      1. A forward hook on ln_f must fire during forward (proves it is called).
      2. Perturbing ln_f's affine params must move the logits (proves its output
         flows into the head, not a dead/ignored module).
    """
    cfg = _tiny_config()
    model = GPT2(cfg)
    model.eval()

    assert isinstance(model.ln_f, torch.nn.LayerNorm)

    fired = {"called": False}
    handle = model.ln_f.register_forward_hook(lambda *a: fired.__setitem__("called", True))
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        logits_before = model(idx)
    handle.remove()
    assert fired["called"], "ln_f was never applied in the forward path"

    # ln_f must genuinely feed the head: changing its affine params moves logits.
    with torch.no_grad():
        model.ln_f.weight.mul_(3.0)
        model.ln_f.bias.add_(1.0)
        logits_after = model(idx)
    assert not torch.allclose(logits_before, logits_after, atol=1e-6), (
        "perturbing ln_f did not change logits -> ln_f does not feed the lm_head"
    )


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
