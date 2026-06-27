"""Tests for the from-scratch byte-level BPE tokenizer."""

import ast
import inspect

import pytest

from gpt2.tokenizer import BPETokenizer


CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox is quick and the dog is lazy. "
) * 50


def test_train_produces_requested_vocab_size():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    # 256 byte tokens + learned merges == vocab_size when the corpus is rich enough.
    assert len(tok.vocab) == 300
    assert len(tok.merges) == 300 - 256


def test_train_stops_early_when_corpus_exhausted():
    """If the corpus runs out of mergeable pairs, training stops early
    (vocab is approximately, not exactly, the requested size) rather than erroring."""
    tok = BPETokenizer()
    tok.train("ab ab ab", vocab_size=400)
    assert 256 <= len(tok.vocab) < 400
    # and it must still round-trip.
    assert tok.decode(tok.encode("ab ab")) == "ab ab"


def test_vocab_size_below_256_is_rejected():
    tok = BPETokenizer()
    with pytest.raises(ValueError):
        tok.train(CORPUS, vocab_size=255)


def test_vocab_size_exactly_256_learns_no_merges():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=256)
    assert len(tok.merges) == 0
    assert len(tok.vocab) == 256


def test_encode_returns_list_of_ints():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    ids = tok.encode("the quick dog")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_merges_actually_compress():
    """A correct BPE must apply learned merges, not just emit raw bytes.

    A trivial (wrong) impl that returns list(text.encode()) would round-trip
    but never compress, so this distinguishes right from wrong.
    """
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    text = "the quick brown fox"
    ids = tok.encode(text)
    raw_byte_count = len(text.encode("utf-8"))
    assert len(ids) < raw_byte_count


def test_roundtrip_lossless_ascii():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    text = "the lazy dog jumps"
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_lossless_unicode_unseen():
    """Byte-level BPE must round-trip arbitrary unicode never seen in training."""
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)  # ASCII-only corpus
    text = "héllo 世界 🚀 café — naïve"
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert tok.decode(ids) == text


def test_roundtrip_empty_string():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    assert tok.decode(tok.encode("")) == ""


def test_no_third_party_tokenizer_imported():
    """Pure Python (+NumPy/PyTorch); no HF / tokenizers library.

    Parse the actual import statements (not a substring scan, which would
    trip on docstrings) and assert no banned module is imported.
    """
    import gpt2.tokenizer as mod

    banned = {"tokenizers", "transformers", "huggingface_hub", "tiktoken", "sentencepiece"}
    tree = ast.parse(inspect.getsource(mod))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert banned.isdisjoint(imported_roots), f"banned imports present: {banned & imported_roots}"


def test_train_accepts_iterable_of_strings():
    tok = BPETokenizer()
    tok.train(["the quick brown fox ", "jumps over the lazy dog "] * 50, vocab_size=300)
    text = "the lazy fox"
    assert tok.decode(tok.encode(text)) == text
