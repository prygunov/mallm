from __future__ import annotations

import os
import asyncio
import re
from typing import List, Tuple, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

from tools.open_url import open_url_tool
from tools.duck_search import ddg_search_tool
from tools.ask_human import ask_human_tool
from tools.browser_use import browser_tool
from tools.google_search import google_search_tool
from tools.calculate_tool import calculate_tool
from tools.string_tools import before_tool

# Load env and initialize LLMs
load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    planner_llm = ChatOpenAI(model="gpt-4o-mini")
    agent_llm = ChatOpenAI(model="gpt-4o")
else:
    planner_llm = agent_llm = None

# Tools
TOOLS = [
    calculate_tool,
    browser_tool,
    google_search_tool,
    open_url_tool,
]
TOOLS = [t for t in TOOLS if t]

# Agent for executing individual tasks
if agent_llm:
    react_prompt = PromptTemplate.from_file("prompts/react_prompt.txt")
    react_agent = create_react_agent(agent_llm, TOOLS, react_prompt)
    executor = AgentExecutor(agent=react_agent, tools=TOOLS, verbose=True, handle_parsing_errors=True)
else:
    executor = None

MAX_STEPS = 20
plan_prompt = PromptTemplate.from_file("prompts/plan_prompt.txt")
replan_prompt = PromptTemplate.from_file("prompts/replan_prompt.txt")


def ask_planner(prompt_text: str) -> List[str]:
    response = planner_llm.invoke(prompt_text)
    lines = response.content.splitlines()
    return [re.sub(r"^\d+[.)]\s*", "", ln).strip() for ln in lines if ln.strip()]


def initial_plan(query: str) -> List[str]:
    return ask_planner(plan_prompt.format(input=query, tools=TOOLS))


def replan(query: str, completed: List[Tuple[str, str]]) -> List[str]:
    completed_block = "\n".join(f"- {t}: {r}" for t, r in completed) or "(none)"
    prompt_text = replan_prompt.format(tools=TOOLS, input=query, completed_block=completed_block)
    return ask_planner(prompt_text)


class AgentState(TypedDict, total=False):
    query: str
    tasks: List[str]
    completed: List[Tuple[str, str]]
    step: int


async def plan(state: AgentState) -> AgentState:
    tasks = initial_plan(state["query"])
    return {"tasks": tasks, "completed": [], "step": 0}


async def execute(state: AgentState) -> AgentState:
    task = state["tasks"][0]
    facts = "\n".join(f"{t} - {r}" for t, r in state["completed"])
    agent_input = f"Current task: {task}\nFacts: {facts}"
    result = await executor.ainvoke({"input": agent_input})
    output = result["output"]
    state["completed"].append((task, output))
    state["tasks"] = state["tasks"][1:]
    state["step"] += 1
    return state


def update_plan(state: AgentState) -> AgentState:
    state["tasks"] = replan(state["query"], state["completed"])
    return state


def should_continue(state: AgentState) -> str:
    if not state["tasks"] or state["step"] >= MAX_STEPS:
        return END
    return "execute"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("replan", update_plan)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "replan")
    graph.add_conditional_edges("replan", should_continue, {"execute": "execute", "end": END})
    return graph


def run(query: str) -> None:
    graph = build_graph().compile()
    inputs = {"query": query}
    for output in graph.stream(inputs):
        if "tasks" in output:
            print("Tasks:", output["tasks"])
        if "completed" in output:
            last = output["completed"][-1]
            print(f"Step {output['step']}: {last[0]} -> {last[1]}")


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What is 2 + 2?"
    asyncio.run(run(q))
