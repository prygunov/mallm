from __future__ import annotations
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from tools.calculate_tool import calculate_tool
from tools.ltm_tool import ltm_search_tool
from shared_memory import shared_memory

load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    agent_llm = ChatOpenAI(model="gpt-4o")
else:
    agent_llm = None

TOOLS = [calculate_tool, ltm_search_tool]
TOOLS = [t for t in TOOLS if t]

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

async def run_calculator(task: str) -> str:
    """Run the calculator agent with the provided task. It evaluates a mathematical expressions."""
    if not _executor:
        raise RuntimeError("LLM is not configured")
    context = shared_memory.get_context()
    agent_input = f"Context:\n{context}\n\nTask: {task}" if context else task
    result = await _executor.ainvoke({"input": agent_input})
    output = result.get("output", "")
    shared_memory.add(f"Calculator: {task}\n{output}")
    return output

calculator_agent_tool = StructuredTool.from_function(
    name="use_calculator_agent",
    coroutine=run_calculator,
)
