"""A from-scratch byte-level Byte-Pair Encoding (BPE) tokenizer.

Pure Python — no Hugging Face, no ``tokenizers`` library. Operates over raw
UTF-8 bytes (256 base tokens) so that *arbitrary* text — including unicode and
characters never seen during training — round-trips losslessly, matching the
design of GPT-2's tokenizer.

Usage::

    tok = BPETokenizer()
    tok.train("some training corpus ...", vocab_size=400)
    ids = tok.encode("hello world")
    text = tok.decode(ids)   # == "hello world"
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Tuple, Union

Pair = Tuple[int, int]


def _get_stats(ids: List[int]) -> Dict[Pair, int]:
    """Count occurrences of every adjacent pair of token ids."""
    counts: Dict[Pair, int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids: List[int], pair: Pair, new_id: int) -> List[int]:
    """Replace every occurrence of ``pair`` in ``ids`` with ``new_id``."""
    out: List[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE tokenizer with in-memory ``train`` / ``encode`` / ``decode``."""

    def __init__(self) -> None:
        # learned merges: (id, id) -> new_id, in the order they were learned.
        self.merges: Dict[Pair, int] = {}
        # id -> bytes; the 256 single-byte tokens are always present.
        self.vocab: Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

    def train(
        self,
        corpus: Union[str, Iterable[str]],
        vocab_size: int,
        verbose: bool = False,
    ) -> None:
        """Learn merge rules from ``corpus`` up to the target ``vocab_size``.

        ``vocab_size`` must be at least 256 (the number of base byte tokens).
        Training is idempotent: it resets any previously learned state.
        """
        if vocab_size < 256:
            raise ValueError(
                f"vocab_size must be >= 256 (got {vocab_size}); 256 byte tokens are the base."
            )

        text = corpus if isinstance(corpus, str) else "".join(corpus)

        # reset to base state so repeated train() calls are deterministic.
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256

        for i in range(num_merges):
            stats = _get_stats(ids)
            if not stats:
                # corpus exhausted of mergeable pairs before reaching vocab_size.
                if verbose:
                    print(f"stopping early at {256 + i} tokens: no pairs left to merge")
                break
            # pick the most frequent pair; ties broken by pair order for determinism.
            pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
            new_id = 256 + i
            ids = _merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {new_id}")

    def encode(self, text: str) -> List[int]:
        """Encode ``text`` into a list of integer token ids."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = _get_stats(ids)
            # find the pair with the lowest merge index (earliest-learned merge).
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing left that can be merged
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token ids back into text (inverse of ``encode``)."""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    # ---- persistence -----------------------------------------------------

    # Bump if the on-disk format ever changes incompatibly.
    SCHEMA_VERSION = 1

    def save(self, path: str) -> None:
        """Persist the learned merge rules to ``path`` as human-readable JSON.

        The merges are the complete source of truth: the full ``vocab`` is
        deterministically rebuilt from the 256 base byte tokens by replaying
        the merges in order, so only the merges need to be stored. Each merge
        is written as ``[first_id, second_id, new_id]`` in learned order, which
        keeps the file both readable and exactly reconstructable.
        """
        merges = [
            [int(pair[0]), int(pair[1]), int(new_id)]
            for pair, new_id in self.merges.items()
        ]
        # learned order matters for encode/decode; persist it deterministically.
        merges.sort(key=lambda m: m[2])
        payload = {"version": self.SCHEMA_VERSION, "merges": merges}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Reconstruct a tokenizer from a file written by :meth:`save`.

        The returned tokenizer's ``encode``/``decode`` behave identically to
        the instance that was saved.
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        version = payload.get("version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported tokenizer file version {version!r}; "
                f"expected {cls.SCHEMA_VERSION}"
            )

        tok = cls()
        for first, second, new_id in payload["merges"]:
            pair = (int(first), int(second))
            new_id = int(new_id)
            tok.merges[pair] = new_id
            tok.vocab[new_id] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
        return tok


def _demo() -> None:
    """Train on a tiny inline corpus and print a sample encode/decode."""
    corpus = (
        "the quick brown fox jumps over the lazy dog. "
        "a byte-pair encoder learns merges from raw bytes. "
    ) * 20

    tok = BPETokenizer()
    tok.train(corpus, vocab_size=320)

    sample = "the quick brown fox says héllo 世界 🚀"
    ids = tok.encode(sample)
    restored = tok.decode(ids)

    print(f"trained vocab size : {len(tok.vocab)} ({len(tok.merges)} merges)")
    print(f"sample text        : {sample!r}")
    print(f"encoded ({len(ids)} ids)  : {ids}")
    print(f"decoded            : {restored!r}")
    print(f"lossless round-trip: {restored == sample}")


if __name__ == "__main__":
    _demo()
