from langchain_core.tools import StructuredTool

def calculate(what: str) -> str:
    """Calculate a mathematical expression. Input should be a valid Python expression."""
    try:
        return str(eval(what))
    except Exception as e:
        return f"Error in calculate: {e}"

calculate_tool = StructuredTool.from_function(name="calculate_expression", coroutine=calculate)