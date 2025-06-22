try:
    from browser_use import (
        Agent as BrowserAgent,
        BrowserSession,
        BrowserProfile,
        Controller,
        ActionResult,
    )
except ImportError:
    BrowserAgent = BrowserSession = BrowserProfile = Controller = ActionResult = None

import os
import tempfile
from langchain_core.tools import StructuredTool
try:
    from crewai.tools.structured_tool import CrewStructuredTool
except ImportError:
    CrewStructuredTool = None
from patchright.async_api import async_playwright
import openai
from langchain_openai import ChatOpenAI

if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini")
else:
    llm = None

controller = Controller() if Controller else None
if controller:
    @controller.action("Ask user for information")
    def ask_human(question: str) -> str:
        answer = input(f"\n{question}\nInput: ")
        return ActionResult(extracted_content=answer)

if BrowserProfile:
    profile = BrowserProfile(
        channel="chromium",
        keep_alive=True,
        headless=False,
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/136.0.0.0 Safari/537.36"
        ),
        ignore_default_args=["--enable-automation", "--disable-extensions"],
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
        user_data_dir=tempfile.mkdtemp(prefix="bu_tmp_"),
        locale="ru-RU",
        cookies_file="cf_cookies.json",
    )
else:
    profile = None


async def browse(task: str) -> str:
    """Navigate to sites with a browser and perform actions."""
    async with async_playwright() as pw:
        session = BrowserSession(playwright=pw, browser_profile=profile)
        agent = BrowserAgent(task=task, llm=llm, browser_session=session, controller=controller)
        result = await agent.run()
        if result.is_done():
            return result.final_result()
        content = ""
        for r in result.action_results():
            content += r.extracted_content.strip() + "\n"
        return content

if not BrowserAgent:
    browser_tool = None
else:
    browser_tool = StructuredTool.from_function(name="navigate_browser", coroutine=browse)
    if CrewStructuredTool:
        crew_browser_tool = CrewStructuredTool.from_function(browse, name="navigate_browser")
    else:
        crew_browser_tool = None

