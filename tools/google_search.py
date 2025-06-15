from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchRun

try:
    google_search_tool = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper())
except Exception:  # pragma: no cover - allow missing API key
    google_search_tool = None