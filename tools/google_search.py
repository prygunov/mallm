from langchain_core.tools import StructuredTool
from langchain_google_community import GoogleSearchAPIWrapper

async def search_google(query: str) -> str:
    """Return the five results for a Google search."""
    wrapper = GoogleSearchAPIWrapper()
    results = wrapper.results(query, num_results=5)
    if not results:
        return "No results found"
    content = ""
    for r in results:
        content += r["title"].strip() + "\n"
        content += r["snippet"].strip() + "\n"
        content += r["link"].strip() + "\n\n"
    return content

google_search_tool = StructuredTool.from_function(name="search_google", coroutine=search_google)
