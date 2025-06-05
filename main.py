from __future__ import annotations
import asyncio
import os
import re
import tempfile
from typing import List, Tuple, Set

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchRun
from browser_use import (
    Agent as BrowserAgent,
    BrowserSession,
    BrowserProfile,
    Controller,
    ActionResult,
)
from patchright.async_api import async_playwright
import openai
import nest_asyncio


load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "false"
openai.api_key = os.getenv("OPENAI_API_KEY")

controller = Controller()

planner_llm = ChatOpenAI(model="o3")
llm = ChatOpenAI(model="gpt-4o")


@controller.action("Ask user for information")
def ask_human(question: str) -> str:
    answer = input(f"\n{question}\nInput: ")
    return ActionResult(extracted_content=answer)


COOKIES = "cf_cookies.json"
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


def calculate(what: str) -> str:
    print(f"Calculating: {what}")
    try:
        return str(eval(what))
    except Exception as e:  # pragma: no cover - evaluation errors shown to user
        return f"Error in calculate: {e}"


# LangChain agent setup
nest_asyncio.apply()
rl = InMemoryRateLimiter(requests_per_second=0.3)
agent_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", rate_limiter=rl, api_key=os.environ["OPENAI_API_KEY"])
prompt = hub.pull("hwchase17/react")

search_tool = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper())
browser_tool = StructuredTool.from_function(name="navigate_browser", coroutine=browse)
tools = [browser_tool, search_tool]

agent = create_react_agent(agent_llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

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
        else:
            print("✅ Запрос выполнен. Все задачи закрыты.")
            break
    else:
        if step >= MAX_STEPS:
            print("⚠️ Достигнут лимит шагов. Останавливаемся, чтобы избежать бесконечного цикла.")


if __name__ == "__main__":
    asyncio.run(main())
