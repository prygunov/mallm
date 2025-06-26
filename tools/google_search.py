from langchain_core.tools import StructuredTool
try:
    from crewai.tools.structured_tool import CrewStructuredTool
except ImportError:
    CrewStructuredTool = None
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
if CrewStructuredTool:
    crew_google_search_tool = CrewStructuredTool.from_function(search_google, name="search_google")
else:
    crew_google_search_tool = None

