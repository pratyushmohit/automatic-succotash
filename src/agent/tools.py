from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_datetime() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result.

    Args:
        expression: A math expression to evaluate, e.g. '2 + 2' or '(10 * 3) / 4'.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"