
from langchain_core.tools import StructuredTool
try:
    from crewai.tools.structured_tool import CrewStructuredTool
except ImportError:
    CrewStructuredTool = None
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

async def search_duckduckgo(query: str) -> str:
    """Return the five results for a DuckDuckGo search."""
    wrapper = DuckDuckGoSearchAPIWrapper()
    results = wrapper.results(query, max_results=5)
    if not results:
        return "No results found"

    content = ""
    for r in results:
        content += r["title"].strip() + "\n"
        content += r["snippet"].strip() + "\n"
        content += r["link"].strip() + "\n\n"

    return content

ddg_search_tool = StructuredTool.from_function(name="search_duckduckgo", coroutine=search_duckduckgo)
if CrewStructuredTool:
    crew_ddg_search_tool = CrewStructuredTool.from_function(search_duckduckgo, name="search_duckduckgo")
else:
    crew_ddg_search_tool = None
