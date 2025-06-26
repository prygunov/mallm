from langchain_core.tools import StructuredTool
from long_term_memory import long_term_memory

async def search_ltm(query: str) -> str:
    """Search the long-term memory for entries relevant to the query."""
    results = long_term_memory.search(query, k=5)
    if not results:
        return "No relevant memory found"
    return "\n".join(results)

ltm_search_tool = StructuredTool.from_function(
    name="search_ltm",
    coroutine=search_ltm,
)

