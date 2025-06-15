from langchain_core.tools import StructuredTool

async def ask_human(question: str) -> str:
    """Ask the user for information. The user will be prompted to provide an answer."""
    answer = input(f"\n{question}\nInput: ")
    return answer

ask_human_tool = StructuredTool.from_function(name="ask_human", coroutine=ask_human)