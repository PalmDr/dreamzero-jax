"""Compatibility shim for Flax NNX API differences.

Flax 0.10.x does not have ``nnx.List``. This module provides a minimal
replacement that works with NNX graph traversal.
"""

from __future__ import annotations

from typing import Iterator

from flax import nnx


class List(nnx.Module):
    """A list-like container of ``nnx.Module`` instances.

    Registers each element as a numbered attribute so NNX graph traversal
    (``nnx.state``, ``nnx.split``, etc.) discovers them.
    """

    def __init__(self, items: list | None = None):
        self._length = 0
        if items:
            for item in items:
                self.append(item)

    def append(self, item) -> None:
        setattr(self, str(self._length), item)
        self._length += 1

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = self._length + idx
        return getattr(self, str(idx))

    def __setitem__(self, idx: int, value):
        if idx < 0:
            idx = self._length + idx
        setattr(self, str(idx), value)

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator:
        for i in range(self._length):
            yield getattr(self, str(i))
