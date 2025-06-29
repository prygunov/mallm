from __future__ import annotations
import asyncio
import os
import re
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from tools.ask_human import ask_human_tool
from tools.ltm_tool import ltm_search_tool
from agents.calculator_agent import calculator_agent_tool
from agents.search_agent import search_agent_tool
from agents.reasoner_agent import reasoner_agent_tool
from shared_memory import shared_memory

load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    planner_llm = ChatOpenAI(model="gpt-4o-mini")
    agent_llm = ChatOpenAI(model="gpt-4o")
else:
    planner_llm = agent_llm = None

MAX_STEPS = 20

AVAILABLE_TOOLS = [
    calculator_agent_tool,
    search_agent_tool,
    reasoner_agent_tool,
    ltm_search_tool,
    ask_human_tool,
]
AVAILABLE_TOOLS = [t for t in AVAILABLE_TOOLS if t]

if agent_llm:
    prompt = PromptTemplate.from_file("prompts/react_prompt.txt")
    _agent = create_react_agent(agent_llm, AVAILABLE_TOOLS, prompt)
    _executor = AgentExecutor(
        agent=_agent,
        tools=AVAILABLE_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
    )
else:
    _executor = None

plan_prompt = PromptTemplate.from_file("prompts/plan_prompt.txt")
replan_prompt = PromptTemplate.from_file("prompts/replan_prompt.txt")


def _ask_planner(prompt_text: str) -> List[str]:
    if not planner_llm:
        return []
    response = planner_llm.invoke(prompt_text)
    lines = response.content.splitlines()
    return [re.sub(r"^\d+[.)]\s*", "", ln).strip() for ln in lines if ln.strip()]


def initial_plan(query: str) -> List[str]:
    return _ask_planner(plan_prompt.format(input=query, tools=AVAILABLE_TOOLS))


def replan(query: str, completed: List[Tuple[str, str]]) -> List[str]:
    completed_block = "\n".join(f"- {t}: {r}" for t, r in completed) or "(none)"
    prompt_text = replan_prompt.format(
        tools=AVAILABLE_TOOLS, input=query, completed_block=completed_block
    )
    return _ask_planner(prompt_text)


async def run(query: str) -> str:
    if not _executor:
        raise RuntimeError("LLM is not configured")
    shared_memory.add(f"User query: {query}")
    tasks = initial_plan(query)
    print("Planned tasks:")
    print(tasks)
    completed: List[Tuple[str, str]] = []
    step = 0
    while tasks and step < MAX_STEPS:
        step += 1
        current_task = tasks.pop(0)
        facts = "\n".join(f"{k} - {v}" for k, v in completed)
        context = shared_memory.get_context()
        agent_input = (
            f"Context:\n{context}\n\nCurrent task: {current_task}\nFacts: {facts}"
            if context
            else f"Current task: {current_task}\nFacts: {facts}"
        )
        result = await _executor.ainvoke({"input": agent_input})
        output = result.get("output", "")
        completed.append((current_task, output))
        shared_memory.add(f"{current_task} -> {output}")
        tasks = replan(query, completed)
        if tasks and len(tasks) == 1 and tasks[0] == "Nothing.":
            break
    if completed:
        return completed[-1][1]
    return ""
