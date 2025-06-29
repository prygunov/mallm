from __future__ import annotations
"""Reasoner agent using the `o3` model.

This agent draws conclusions only from facts available in the shared context
memory. It does not access external tools or additional information."""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from shared_memory import shared_memory

load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    agent_llm = ChatOpenAI(model="o3")
else:
    agent_llm = None

TOOLS: list[StructuredTool] = []

if agent_llm:
    prompt = PromptTemplate.from_file("prompts/react_prompt.txt")
    _agent = create_react_agent(agent_llm, TOOLS, prompt)
    _executor = AgentExecutor(
        agent=_agent,
        tools=TOOLS,
        verbose=False,
        handle_parsing_errors=True,
    )
else:
    _executor = None

async def run_reasoner(task: str) -> str:
    """Run the reasoner agent with the provided task. It draws conclusions based on the shared context."""
    if not _executor:
        raise RuntimeError("LLM is not configured")
    context = shared_memory.get_context()
    agent_input = f"Context:\n{context}\n\nTask: {task}" if context else task
    result = await _executor.ainvoke({"input": agent_input})
    output = result.get("output", "")
    shared_memory.add(f"Reasoner: {task}\n{output}")
    return output

reasoner_agent_tool = StructuredTool.from_function(
    name="use_reasoner_agent",
    coroutine=run_reasoner,
)
