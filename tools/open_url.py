import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import StructuredTool
from readability import Document
import re

async def open_url(url: str) -> str:
    """Fetch a URL and return reader-mode plain text using httpx."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPError:
        # Retry without SSL verification on certificate errors
        return " forbidden, try another url"
    html = resp.text
    doc = Document(html)
    main_html = doc.summary()
    soup = BeautifulSoup(main_html, "lxml")
    text = soup.get_text(separator="\n")
    result = re.sub(r'(?:\r?\n){3,}', '\n\n', text)

    return result

open_url_tool = StructuredTool.from_function(name="open_url", coroutine=open_url)