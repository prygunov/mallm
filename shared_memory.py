from __future__ import annotations
from typing import List

from long_term_memory import long_term_memory

from langgraph.store.memory import InMemoryStore


class SharedMemory:
    """Shared context memory backed by ``InMemoryStore``.

    Each entry is also appended to the long-term memory on disk.
    """

    def __init__(self, max_length: int = 50) -> None:
        self._store = InMemoryStore()
        self._max_length = max_length
        self._counter = 0

    def add(self, text: str) -> None:
        """Append a new entry to memory."""
        self._store.put(("context",), str(self._counter), {"text": text})
        long_term_memory.add(text)
        self._counter += 1
        if self._counter > self._max_length:
            old_key = str(self._counter - self._max_length - 1)
            self._store.delete(("context",), old_key)

    def get_context(self, n: int | None = None) -> str:
        """Return the last ``n`` entries joined as a single string."""
        if n is None or n > self._max_length:
            n = self._max_length
        start_idx = max(0, self._counter - n)
        texts: List[str] = []
        for idx in range(start_idx, self._counter):
            item = self._store.get(("context",), str(idx))
            if item:
                value = item.value
                texts.append(value.get("text") if isinstance(value, dict) else str(value))
        return "\n".join(texts)


shared_memory = SharedMemory()
