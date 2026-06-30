"""Minimal embedding lookup layer for Chapter 5 examples."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleEmbedding(nn.Module):
    """A small wrapper around ``nn.Embedding`` for teaching token lookup.

    ``input_ids`` with shape ``(B, T)`` become embedding vectors with shape
    ``(B, T, C)`` where ``C`` is ``hidden_size``.
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    @property
    def weight(self) -> torch.nn.Parameter:
        """Expose the lookup table for tests and side-by-side examples."""
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        return self.embedding(input_ids)
