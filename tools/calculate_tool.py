from langchain_core.tools import StructuredTool
try:
    from crewai.tools.structured_tool import CrewStructuredTool
except ImportError:
    CrewStructuredTool = None

def calculate(what: str) -> str:
    """Calculate a mathematical expression. Input should be a valid Python expression."""
    try:
        return str(eval(what))
    except Exception as e:
        return f"Error in calculate: {e}"

calculate_tool = StructuredTool.from_function(name="calculate_expression", coroutine=calculate)
if CrewStructuredTool:
    crew_calculate_tool = CrewStructuredTool.from_function(calculate, name="calculate_expression")
else:
    crew_calculate_tool = None

