from __future__ import annotations
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from tools.google_search import google_search_tool
from tools.open_url import open_url_tool

load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    agent_llm = ChatOpenAI(model="gpt-4o")
else:
    agent_llm = None

TOOLS = [google_search_tool, open_url_tool]
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

async def run_search(task: str) -> str:
    if not _executor:
        raise RuntimeError("LLM is not configured")
    result = await _executor.ainvoke({"input": task})
    return result.get("output", "")

search_agent_tool = StructuredTool.from_function(
    name="use_search_agent",
    coroutine=run_search,
)
