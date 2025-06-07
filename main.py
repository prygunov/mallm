from __future__ import annotations
import asyncio
import os
import re
import tempfile
from typing import List, Tuple, Set

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chains.pebblo_retrieval.models import Prompt
from langchain_core.prompts import PromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from open_url import open_url
try:
    from browser_use import (
        Agent as BrowserAgent,
        BrowserSession,
        BrowserProfile,
        Controller,
        ActionResult,
    )
except ImportError:  # pragma: no cover - optional dependency
    BrowserAgent = BrowserSession = BrowserProfile = Controller = ActionResult = None
from patchright.async_api import async_playwright
import openai
import nest_asyncio


load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "false"
openai.api_key = os.getenv("OPENAI_API_KEY")

controller = Controller() if Controller else None

if os.getenv("OPENAI_API_KEY"):
    planner_llm = ChatOpenAI(model="o1")
    llm = ChatOpenAI(model="gpt-4o")
else:  # pragma: no cover - optional during testing
    planner_llm = llm = None


if controller:
    @controller.action("Ask user for information")
    def ask_human(question: str) -> str:
        answer = input(f"\n{question}\nInput: ")
        return ActionResult(extracted_content=answer)


COOKIES = "cf_cookies.json"
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
        cookies_file=COOKIES,
    )
else:
    profile = None


async def browse(task: str) -> str:
    """Navigate to sites with a browser and perform actions."""
    if not BrowserAgent:
        return "browser_use not installed"
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


def calculate(what: str) -> str:
    print(f"Calculating: {what}")
    try:
        return str(eval(what))
    except Exception as e:  # pragma: no cover - evaluation errors shown to user
        return f"Error in calculate: {e}"


async def search_duckduckgo(query: str) -> str:
    """Return the first result link for a DuckDuckGo search."""
    wrapper = DuckDuckGoSearchAPIWrapper()
    results = wrapper.results(query, max_results=1)
    if not results:
        return "No results found"
    return results[0]["link"]

# LangChain agent setup
nest_asyncio.apply()
rl = InMemoryRateLimiter(requests_per_second=0.3)
if os.getenv("OPENAI_API_KEY"):
    agent_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", rate_limiter=rl, api_key=os.environ["OPENAI_API_KEY"])
else:  # pragma: no cover - optional during testing
    agent_llm = None

from langsmith import Client
client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
prompt = client.pull_prompt("hwchase17/react", include_model=True)

try:
    search_tool = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper())
except Exception:  # pragma: no cover - allow missing API key
    search_tool = None

browser_tool = StructuredTool.from_function(name="navigate_browser", coroutine=browse)
ddg_search_tool = StructuredTool.from_function(name="search_duckduckgo", coroutine=search_duckduckgo)
open_url_tool = StructuredTool.from_function(name="open_url", coroutine=open_url)
tools = [tool for tool in (browser_tool, search_tool, ddg_search_tool, open_url_tool) if tool]

if agent_llm:
    agent = create_react_agent(agent_llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
else:  # pragma: no cover - optional during testing
    agent = agent_executor = None

MAX_STEPS = 20

PLANNER_SYSTEM_MSG = (
    "You are an expert project planner. Given a user request, break it down into an ordered list of atomic tasks. "
    "Return **only** the list, each task on a new line, numbered."
)
REPLAN_SYSTEM_MSG = (
    "You are an expert project planner. The user's original request is provided below. "
    "Some tasks have already been completed. "
    "Given the completed tasks and their outcomes, return an ordered list of remaining atomic tasks needed to fully satisfy the request. "
    "Do not repeat completed tasks. Return ONLY the list, each task on a new line, numbered. If nothing remains, return nothing."
)


def ask_planner(prompt_text: str) -> List[str]:
    response = planner_llm.invoke(prompt_text)
    lines = response.content.splitlines()
    return [re.sub(r"^\d+[.)]\s*", "", ln).strip() for ln in lines if ln.strip()]


def initial_plan(query: str) -> List[str]:
    return ask_planner(PLANNER_SYSTEM_MSG + f"\n\nUser request:\n{query}\n\nTasks:")


def replan(query: str, completed: List[Tuple[str, str]]) -> List[str]:
    completed_block = "\n".join(f"- {t}: {r}" for t, r in completed) or "(none)"
    prompt_text = REPLAN_SYSTEM_MSG + f"\n\nUser request:\n{query}\n\nCompleted tasks and results:\n{completed_block}\n\nRemaining tasks:"
    return ask_planner(prompt_text)


aSYNC_QUERY = """What is the last word before the second chorus of the King of Pop's fifth single from his sixth studio album?"""


async def main(query: str = aSYNC_QUERY) -> None:
    print("Получено задание:", query)
    tasks = initial_plan(query)
    completed: List[Tuple[str, str]] = []
    step = 0
    while tasks and step < MAX_STEPS:
        step += 1
        current_task = tasks.pop(0)
        print(f"===== 🔹 Step {step}: {current_task} 🔹 =====")
        try:
            facts = "\n".join(f"{k} - {v}" for k, v in completed)
            agent_input = f"Current task: {current_task}\nFacts: {facts}"
            result = await agent_executor.ainvoke({"input": agent_input}, return_only_outputs=True)
            output = result["output"]
            await asyncio.sleep(2.5)
            print("🔸 Ответ:", output, "\n")
        except Exception as e:  # pragma: no cover - runtime errors printed
            output = f"ERROR: {e}"
            print("⚠️ Ошибка:", e, "\n")
        completed.append((current_task, output))
        tasks = replan(query, completed)
        if tasks:
            print("📋 Новый план:\n" + "\n".join(f"  {i+1}. {t}" for i, t in enumerate(tasks)))
            print()
            print("Текущие факты:" + "\n" + "\n".join(f"  {i+1}. {t} - {r}" for i, (t, r) in enumerate(completed)))
        else:
            print("✅ Запрос выполнен. Все задачи закрыты.")
            break
    else:
        if step >= MAX_STEPS:
            print("⚠️ Достигнут лимит шагов. Останавливаемся, чтобы избежать бесконечного цикла.")


if __name__ == "__main__":
    asyncio.run(main())
