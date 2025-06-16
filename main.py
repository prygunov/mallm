from __future__ import annotations
import asyncio
import os
import re

from typing import List, Tuple
from dotenv import load_dotenv
import traceback

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
import openai
import nest_asyncio

# Init
load_dotenv()
nest_asyncio.apply()

from tools.open_url import open_url_tool
from tools.duck_search import ddg_search_tool
from tools.ask_human import ask_human_tool
from tools.browser_use import browser_tool
from tools.google_search import google_search_tool
from tools.calculate_tool import calculate_tool
from tools.string_tools import before_tool
from langchain_core.prompts import PromptTemplate

if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    planner_llm = ChatOpenAI(model="gpt-4o-mini")
    agent_llm = ChatOpenAI(model="gpt-4o")
else:
    planner_llm = agent_llm = None

def split_text(text: str, delimiter: str) -> List[str]:
    """Split text by a delimiter and return a list of non-empty parts."""
    parts = [part.strip() for part in text.split(delimiter) if part.strip()]
    return parts

tools_to_use = (
    #ask_human_tool,
    calculate_tool,
    browser_tool,
    google_search_tool,
    #ddg_search_tool,
    open_url_tool,
    # before_tool
)

available_tools = [tool for tool in tools_to_use if tool]

if agent_llm:
    prompt = PromptTemplate.from_file("prompts/react_prompt.txt")
    agent = create_react_agent(agent_llm, available_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=available_tools, verbose=True, handle_parsing_errors=True)
else:
    agent = agent_executor = None

MAX_STEPS = 20

plan_prompt = PromptTemplate.from_file("prompts/plan_prompt.txt")
replan_prompt = PromptTemplate.from_file("prompts/replan_prompt.txt")

def ask_planner(prompt_text: str) -> List[str]:
    response = planner_llm.invoke(prompt_text)
    lines = response.content.splitlines()
    return [re.sub(r"^\d+[.)]\s*", "", ln).strip() for ln in lines if ln.strip()]

def initial_plan(query: str) -> List[str]:
    return ask_planner(plan_prompt.format(input=query, tools = available_tools))

def replan(query: str, completed: List[Tuple[str, str]]) -> List[str]:
    completed_block = "\n".join(f"- {t}: {r}" for t, r in completed) or "(none)"
    prompt_text = replan_prompt.format(tools = available_tools, input = query, completed_block = completed_block)
    return ask_planner(prompt_text)


query = """What is the last word before the second chorus of the King of Pop's fifth single from his sixth studio album?"""
# query = """Open url https://www.azlyrics.com/lyrics/michaeljackson/humannature.html and get last word before the second chorus"""#  todo substring tool
# query = "что делает management: observations: long-task-timer: enabled: false в Spring Boot 3"

# query = "Hi there"

async def main(query: str = query) -> None:
    print("Получено задание:", query)
    print("Доступные инструменты:", ", ".join(tool.name for tool in available_tools))
    tasks = initial_plan(query)
    print("📋 Новый план:\n" + "\n".join(f"  {i + 1}. {t}" for i, t in enumerate(tasks)))
    completed: List[Tuple[str, str]] = []
    step = 0
    while tasks and step < MAX_STEPS:
        step += 1
        current_task = tasks.pop(0)
        print(f"===== 🔹 Step {step}: {current_task} 🔹 =====")
        try:
            facts = "\n".join(f"{k} - {v}" for k, v in completed)
            agent_input = f"Current task: {current_task}\nFacts: {facts}"
            result = await agent_executor.ainvoke({"input": agent_input})
            output = result["output"]
            # await asyncio.sleep(2.5)
            print("🔸 Ответ:", output, "\n")
            # todo handle only relevant output with llm

        except Exception as e:
            # Handle any exceptions that occur during task execution
            print(traceback.format_exc())
            output = f"⚠️ Ошибка: {e}"
            print("⚠️ Ошибка:", e, "\n")

        completed.append((current_task, output))

        tasks = replan(query, completed)
        if tasks:
            if len(tasks) == 1 and tasks[0] == "Nothing.":
                break
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
