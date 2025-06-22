import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import StructuredTool
from readability import Document
import re
import uuid

from .memory_cache import cache_text

from urllib.parse import urlparse, urlunparse, quote

def myquote(url):
    parts = urlparse(url)
    return urlunparse(parts._replace(path=quote(parts.path)))

async def open_url(url: str) -> str:
    """Fetch a URL, store reader-mode text in the cache, and return the key."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(myquote(url.strip()))
            resp.raise_for_status()
    except httpx.HTTPError:
        # Retry without SSL verification on certificate errors
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0, verify=False) as client:
            resp = await client.get(url)
            resp.raise_for_status()

    html = resp.text
    doc = Document(html)
    main_html = doc.summary()
    soup = BeautifulSoup(main_html, "lxml")
    text = soup.get_text(separator="\n")
    result = re.sub(r'(?:\r?\n){3,}', '\n\n', text)

    key = uuid.uuid4().hex
    await cache_text(key, result)
    return key

open_url_tool = StructuredTool.from_function(
    name="open_url",
    coroutine=open_url,
    description="Fetch a URL, save reader-mode text in a temporary cache, and return the cache key.",
)
