from __future__ import annotations
import os
import uuid
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class LongTermMemory:
    """Persistent memory stored locally with embeddings for retrieval."""

    def __init__(self, path: str = "ltm_memory.txt", persist_dir: str = "ltm_db") -> None:
        self.path = path
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        # Ensure file exists
        open(self.path, "a", encoding="utf-8").close()
        self._embeddings = None
        if os.getenv("OPENAI_API_KEY"):
            self._embeddings = OpenAIEmbeddings()
        self._store = Chroma(
            collection_name="ltm",
            embedding_function=self._embeddings,
            persist_directory=self.persist_dir,
        )

    def add(self, text: str) -> None:
        """Append a new entry to disk and vector store."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        if self._embeddings:
            self._store.add_texts([text], ids=[str(uuid.uuid4())])
            self._store.persist()

    def get_context(self, n: int | None = None) -> str:
        """Return the last ``n`` entries joined as a single string."""
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        if n is not None:
            lines = lines[-n:]
        return "\n".join(lines)

    def search(self, query: str, k: int = 5) -> List[str]:
        """Return the most similar stored entries to ``query``."""
        if not self._embeddings:
            return []
        results = self._store.similarity_search(query, k=k)
        return [r.page_content for r in results]


long_term_memory = LongTermMemory()
