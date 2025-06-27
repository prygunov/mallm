"""Tool for evaluating mathematical expressions."""

from langchain_core.tools import StructuredTool
import math


# ──────────────────────────── Basic arithmetic ─────────────────────────────
def add(a: float, b: float) -> float:
    """Return ``a + b``."""

    return a + b


def subtract(a: float, b: float) -> float:
    """Return ``a - b``."""

    return a - b


def multiply(a: float, b: float) -> float:
    """Return ``a * b``."""

    return a * b


def divide(a: float, b: float) -> float:
    """Return ``a / b``."""

    return a / b


def int_divide(a: float, b: float) -> float:
    """Return ``a`` floor-divided by ``b``."""

    return a // b


def modulo(a: float, b: float) -> float:
    """Return ``a % b``."""

    return a % b


def power(a: float, b: float) -> float:
    """Return ``a`` raised to ``b``."""

    return a**b


def abs_val(x: float) -> float:
    """Return ``abs(x)``."""

    return abs(x)


def round_val(x: float, ndigits: int | None = None) -> float:
    """Round ``x`` to ``ndigits`` digits."""

    return round(x, ndigits) if ndigits is not None else round(x)


def minimum(a: float, b: float) -> float:
    """Return the lesser of ``a`` and ``b``."""

    return min(a, b)


def maximum(a: float, b: float) -> float:
    """Return the greater of ``a`` and ``b``."""

    return max(a, b)


# ────────────────────────────── Math module ────────────────────────────────
def sqrt(x: float) -> float:
    """Return the square root of ``x``."""

    return math.sqrt(x)


def log(x: float, base: float = math.e) -> float:
    """Return ``log(x, base)``."""

    return math.log(x, base)


def log10(x: float) -> float:
    """Return ``log10(x)``."""

    return math.log10(x)


def log2(x: float) -> float:
    """Return ``log2(x)``."""

    return math.log2(x)


def exp(x: float) -> float:
    """Return ``e`` raised to ``x``."""

    return math.exp(x)


def sin(x: float) -> float:
    """Return ``sin(x)``."""

    return math.sin(x)


def cos(x: float) -> float:
    """Return ``cos(x)``."""

    return math.cos(x)


def tan(x: float) -> float:
    """Return ``tan(x)``."""

    return math.tan(x)


def asin(x: float) -> float:
    """Return ``asin(x)``."""

    return math.asin(x)


def acos(x: float) -> float:
    """Return ``acos(x)``."""

    return math.acos(x)


def atan(x: float) -> float:
    """Return ``atan(x)``."""

    return math.atan(x)


def sinh(x: float) -> float:
    """Return ``sinh(x)``."""

    return math.sinh(x)


def cosh(x: float) -> float:
    """Return ``cosh(x)``."""

    return math.cosh(x)


def tanh(x: float) -> float:
    """Return ``tanh(x)``."""

    return math.tanh(x)


def degrees(x: float) -> float:
    """Convert radians to degrees."""

    return math.degrees(x)


def radians(x: float) -> float:
    """Convert degrees to radians."""

    return math.radians(x)


def factorial(n: int) -> int:
    """Return ``n!``."""

    return math.factorial(n)


def floor(x: float) -> float:
    """Return ``floor(x)``."""

    return math.floor(x)


def ceil(x: float) -> float:
    """Return ``ceil(x)``."""

    return math.ceil(x)


def gcd(a: int, b: int) -> int:
    """Return the greatest common divisor of ``a`` and ``b``."""

    return math.gcd(a, b)


def lcm(a: int, b: int) -> int:
    """Return the least common multiple of ``a`` and ``b``."""

    return math.lcm(a, b)


def hypot(a: float, b: float) -> float:
    """Return ``sqrt(a*a + b*b)``."""

    return math.hypot(a, b)


def permutations(n: int, k: int | None = None) -> int:
    """Return ``nPk`` permutations."""

    return math.perm(n, k) if k is not None else math.perm(n)


def combinations(n: int, k: int) -> int:
    """Return ``nCk`` combinations."""

    return math.comb(n, k)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp ``value`` between ``min_value`` and ``max_value``."""

    return max(min_value, min(value, max_value))


# ──────────────────────────────── Registry ────────────────────────────────
ALLOWED_NAMES = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
    "int_divide": int_divide,
    "modulo": modulo,
    "power": power,
    "abs": abs_val,
    "round": round_val,
    "min": minimum,
    "max": maximum,
    "sqrt": sqrt,
    "log": log,
    "log10": log10,
    "log2": log2,
    "exp": exp,
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "asin": asin,
    "acos": acos,
    "atan": atan,
    "sinh": sinh,
    "cosh": cosh,
    "tanh": tanh,
    "degrees": degrees,
    "radians": radians,
    "factorial": factorial,
    "floor": floor,
    "ceil": ceil,
    "gcd": gcd,
    "lcm": lcm,
    "hypot": hypot,
    "perm": permutations,
    "comb": combinations,
    "clamp": clamp,
    "pi": math.pi,
    "e": math.e,
}

def calculate(what: str) -> str:
    """Evaluate a mathematical expression.

    The expression can use standard arithmetic operators (``+``, ``-``, ``*``,
    ``/``, ``//``, ``%``, ``**``) and any function from ``math`` such as
    ``sin``, ``cos``, ``tan``, ``sqrt``, ``log`` and many others.  Builtin helper
    functions like ``abs`` and ``round`` are also available.
    """
    try:
        result = eval(what, {"__builtins__": {}}, ALLOWED_NAMES)
        return str(result)
    except Exception as e:
        return f"Error in calculate: {e}"

calculate_tool = StructuredTool.from_function(
    name="calculate_expression", coroutine=calculate
)
