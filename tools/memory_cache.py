from typing import Dict
from langchain_core.tools import StructuredTool

_cache: Dict[str, str] = {}

async def cache_text(key: str, text: str) -> str:
    """Store text in an in-memory cache and return a confirmation."""
    _cache[key] = text
    return f"Cached {len(text)} characters under '{key}'."

async def read_cache(key: str) -> str:
    """Return cached text for a key or an empty string if not found."""
    return _cache.get(key, "")

cache_write_tool = StructuredTool.from_function(
    name="cache_write",
    coroutine=cache_text,
    description="Store text in a temporary in-memory cache by key.",
)

cache_read_tool = StructuredTool.from_function(
    name="cache_read",
    coroutine=read_cache,
    description="Read cached text by key. Returns empty string if missing.",
)
