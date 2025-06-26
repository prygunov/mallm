from langchain_core.tools import StructuredTool
try:
    from crewai.tools.structured_tool import CrewStructuredTool
except ImportError:
    CrewStructuredTool = None

async def ask_human(question: str) -> str:
    """Ask the user for information. The user will be prompted to provide an answer."""
    return input(f"\n{question}\nInput: ")

ask_human_tool = StructuredTool.from_function(name="ask_human", coroutine=ask_human)
if CrewStructuredTool:
    crew_ask_human_tool = CrewStructuredTool.from_function(ask_human, name="ask_human")
else:
    crew_ask_human_tool = None
