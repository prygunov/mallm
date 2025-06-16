from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchRun

try:
    google_search_tool = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper())
except Exception:
    google_search_tool = None