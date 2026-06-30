"""Simple teaching-oriented tokenizer for Chapter 5.

This module intentionally uses whitespace tokenization instead of BPE or
SentencePiece so learners can focus on the LLM data flow:

    text -> tokens -> token ids -> embedding vectors
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


class SimpleTokenizer:
    """A tiny whitespace tokenizer with fixed ``<pad>`` and ``<unk>`` tokens.

    The tokenizer is deliberately small and deterministic for textbook examples.
    It is not meant to replace production subword tokenizers.
    """

    pad_token = "<pad>"
    unk_token = "<unk>"

    def __init__(self, tokens: Iterable[str]):
        unique_tokens: list[str] = []
        seen = {self.pad_token, self.unk_token}
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            unique_tokens.append(token)

        self.id_to_token = [self.pad_token, self.unk_token, *unique_tokens]
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}
        self.pad_id = self.token_to_id[self.pad_token]
        self.unk_id = self.token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def tokenize(self, text: str) -> list[str]:
        """Split text by whitespace and drop empty pieces."""
        return text.split()

    def encode(self, text: str) -> list[int]:
        """Convert text into token ids, mapping unseen tokens to ``<unk>``."""
        return [self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text)]

    def decode(self, ids: Sequence[int]) -> list[str]:
        """Convert token ids back to token strings.

        Unknown numeric ids decode to ``<unk>`` so examples stay robust.
        """
        return [self.id_to_token[idx] if 0 <= int(idx) < self.vocab_size else self.unk_token for idx in ids]

    def batch_encode(self, texts: Sequence[str], max_length: int | None = None) -> torch.Tensor:
        """Encode and pad a batch of texts to a rectangular ``LongTensor``.

        If ``max_length`` is omitted, the longest sequence in the batch is used.
        Longer sequences are truncated when ``max_length`` is provided.
        """
        encoded = [self.encode(text) for text in texts]
        if max_length is None:
            max_length = max((len(ids) for ids in encoded), default=0)

        padded = []
        for ids in encoded:
            row = ids[:max_length]
            row = row + [self.pad_id] * (max_length - len(row))
            padded.append(row)
        return torch.tensor(padded, dtype=torch.long)
