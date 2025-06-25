from __future__ import annotations

import os
import asyncio
import re
from typing import List, Tuple, TypedDict
import traceback

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

# Load env and initialize LLMs
load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    planner_llm = ChatOpenAI(model="gpt-4o-mini")
    agent_llm = ChatOpenAI(model="gpt-4o")
    critic_llm = ChatOpenAI(model="gpt-4o-mini")
else:
    planner_llm = agent_llm = critic_llm = None

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
PARALLEL_TASKS = 1
plan_prompt = PromptTemplate.from_file("prompts/plan_prompt.txt")
replan_prompt = PromptTemplate.from_file("prompts/replan_prompt.txt")
critic_prompt = PromptTemplate.from_file("prompts/critic_prompt.txt")

def ask_planner(prompt_text: str) -> List[str]:
    response = planner_llm.invoke(prompt_text)
    lines = response.content.splitlines()
    return [re.sub(r"^\d+[.)]\s*", "", ln).strip() for ln in lines if ln.strip()]

def initial_plan(query: str) -> List[str]:
    return ask_planner(plan_prompt.format(input=query, tools=TOOLS))

def replan(query: str, completed: List[Tuple[str, str]]) -> List[str]:
    print("replannig...")
    completed_block = "\n".join(f"- {t}: {r}" for t, r in completed) or "(none)"
    prompt_text = replan_prompt.format(
        tools=TOOLS,
        input=query,
        completed_block=completed_block,
    )
    return ask_planner(prompt_text)


class AgentState(TypedDict, total=False):
    query: str
    tasks: List[str]
    completed: List[Tuple[str, str]]
    already_tried: List[Tuple[str, str]]
    step: int
    final: str

async def plan(state: AgentState) -> AgentState:
    tasks = initial_plan(state["query"])
    return {"tasks": tasks, "completed": [], "step": 0}


async def execute(state: AgentState) -> AgentState:
    """Execute a batch of tasks in parallel."""
    batch = state["tasks"][:PARALLEL_TASKS]
    if not batch:
        return state

    async def run_task(task: str) -> str:
        facts = "\n".join(f"{t} - {r}" for t, r in state["completed"])
        agent_input = f"Current task: {task}\nFacts: {facts}"
        result = await executor.ainvoke({"input": agent_input})
        return result["output"]

    try:
        outputs = await asyncio.gather(*(run_task(t) for t in batch))

        for task, output in zip(batch, outputs):
            state["completed"].append((task, output))
            state["step"] += 1
    except Exception as e:
        # Handle any exceptions that occur during task execution
        print(traceback.format_exc())
        output = f"⚠️ Ошибка: {e}"
        for task, output in zip(batch, output):
            state["completed"].append((task, output))
            state["step"] += 1
        print("⚠️ Ошибка:", e, "\n")

    state["tasks"] = state["tasks"][len(batch):]
    return state

def update_plan(state: AgentState) -> AgentState:
    state["tasks"] = replan(state["query"], state["completed"])
    return state

async def critique(state: AgentState) -> AgentState:
    completed_block = "\n".join(f"- {t}: {r}" for t, r in state["completed"]) or "(none)"
    print(f"Critique: {completed_block}")
    prompt_text = critic_prompt.format(input=state["query"], completed_block=completed_block)
    response = await critic_llm.ainvoke(prompt_text)
    state["final"] = response.content
    return state


def should_continue(state: AgentState) -> str:
    if state["tasks"][0] == 'Nothing.' or state["step"] >= MAX_STEPS:
        return "end"
    return "execute"


def build_graph() -> StateGraph:
    return (StateGraph(AgentState)
             .add_node("plan", plan)
             .set_entry_point("plan")
             .add_node("execute", execute)
             .add_node("replan", update_plan)
             .add_node("critic", critique)
             .add_edge("plan", "execute")
             .add_edge("execute", "replan")
             .add_conditional_edges("replan", should_continue, {"execute": "execute", "end": "critic"})
             .add_edge("critic", END)
             )

async def run_query(query: str) -> str:
    graph = build_graph().compile()
    inputs = {"query": query}
    seen = 0
    final = ""
    async for event in graph.astream(inputs):
        state = next(iter(event.values()))
        if "tasks" in state:
            print("Tasks:", state["tasks"])
        if "completed" in state:
            new = state["completed"][seen:]
            for i, (task, res) in enumerate(new, start=seen + 1):
                print(f"Step {i}: {task} -> {res}")
            seen += len(new)
        if state.get("final"):
            final = state["final"]

    return final

if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What is the last word before the second chorus of the King of Pop's fifth single from his sixth studio album?"
    answer = asyncio.run(run_query(q))
    print("Final:", answer)

