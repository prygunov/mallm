import asyncio
from main import search_duckduckgo, open_url

async def run():
    url = await search_duckduckgo("langchain open source")
    print("URL:", url)
    if url.startswith("http"):
        text = await open_url(url)
        print(text[:1000])

if __name__ == "__main__":
    asyncio.run(run())
