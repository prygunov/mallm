"""
string_tools.py — набор StructuredTool-ов для манипуляций со строками.

Использование:
    from string_tools import STRING_TOOLS
    agent = initialize_agent(tools=STRING_TOOLS, ...)
"""

from typing import List, Optional
from pydantic import BaseModel, Field

# Совместимость с разными версиями LangChain
try:
    from langchain_core.tools import StructuredTool
except ImportError:  # ≤ 0.0.350
    from langchain.tools import StructuredTool  # type: ignore

try:
    from crewai.tools.structured_tool import CrewStructuredTool
except ImportError:
    CrewStructuredTool = None


# ───────────────────────── helpers ──────────────────────────
def _safe_find(text: str, needle: str) -> int:
    """Возвращает индекс needle или -1, если не найден."""
    try:
        return text.index(needle)
    except ValueError:
        return -1


# ──────────────────────── 1. Before delimiter ─────────────────────────
class BeforeArgs(BaseModel):
    text: str = Field(..., description="Оригинальный текст")
    delimiter: str = Field(
        ..., description="Граница. Берём всё, что левее первой её встречи"
    )
    include_delimiter: bool = Field(
        False, description="Вернуть ли сам delimiter в конце результата"
    )


def get_text_before(text: str, delimiter: str, include_delimiter: bool = False) -> str:
    """Возвращает часть строки до первого появления delimiter."""
    idx = _safe_find(text, delimiter)
    if idx == -1:
        return text
    return text[: idx + (len(delimiter) if include_delimiter else 0)]


before_tool = StructuredTool.from_function(
    func=get_text_before,
    name="text_before_delimiter",
    description=(
        "Вернёт всё, что находится до первой встречи delimiter. "
        "Полезно, когда надо обрезать хвост URI, команды, списка и т.д."
    ),
    args_schema=BeforeArgs,
)
if CrewStructuredTool:
    crew_before_tool = CrewStructuredTool.from_function(
        get_text_before,
        name="text_before_delimiter",
        description="Вернёт всё, что находится до первой встречи delimiter. Полезно, когда надо обрезать хвост URI, команды, списка и т.д.",
        args_schema=BeforeArgs,
    )
else:
    crew_before_tool = None

# ──────────────────────── 2. After delimiter ──────────────────────────
class AfterArgs(BaseModel):
    text: str = Field(..., description="Оригинальный текст")
    delimiter: str = Field(
        ..., description="Граница. Берём всё, что правее первой её встречи"
    )
    include_delimiter: bool = Field(
        False, description="Включить ли delimiter в начало результата"
    )


def get_text_after(text: str, delimiter: str, include_delimiter: bool = False) -> str:
    """Часть строки после первого появления delimiter."""
    idx = _safe_find(text, delimiter)
    if idx == -1:
        return text
    start = idx if include_delimiter else idx + len(delimiter)
    return text[start:]


after_tool = StructuredTool.from_function(
    func=get_text_after,
    name="text_after_delimiter",
    description="Быстро получить «хвост» строки после delimiter.",
    args_schema=AfterArgs,
)
if CrewStructuredTool:
    crew_after_tool = CrewStructuredTool.from_function(
        get_text_after,
        name="text_after_delimiter",
        description="Быстро получить «хвост» строки после delimiter.",
        args_schema=AfterArgs,
    )
else:
    crew_after_tool = None

# ──────────────────────── 3. Between markers ──────────────────────────
class BetweenArgs(BaseModel):
    text: str = Field(..., description="Оригинальный текст")
    start_marker: str = Field(..., description="Левая граница")
    end_marker: str = Field(..., description="Правая граница")
    include_markers: bool = Field(
        False, description="Вернуть ли маркеры вместе с содержимым"
    )


def get_text_between(
    text: str, start_marker: str, end_marker: str, include_markers: bool = False
) -> str:
    """Возвращает подстроку между двумя маркерами (первое-первое совпадение)."""
    start = _safe_find(text, start_marker)
    if start == -1:
        return ""
    start_end = start + len(start_marker)
    end = text.find(end_marker, start_end)
    if end == -1:
        return ""
    if include_markers:
        return text[start : end + len(end_marker)]
    return text[start_end:end]


between_tool = StructuredTool.from_function(
    func=get_text_between,
    name="text_between_markers",
    description="Выцепить фрагмент между start_marker и end_marker.",
    args_schema=BetweenArgs,
)
if CrewStructuredTool:
    crew_between_tool = CrewStructuredTool.from_function(
        get_text_between,
        name="text_between_markers",
        description="Выцепить фрагмент между start_marker и end_marker.",
        args_schema=BetweenArgs,
    )
else:
    crew_between_tool = None

# ──────────────────────── 4. Split & pick ─────────────────────────────
class SplitPickArgs(BaseModel):
    text: str = Field(..., description="Оригинальный текст")
    delimiter: str = Field(..., description="Разделитель для split()")
    index: int = Field(0, description="Какой элемент вернуть (может быть отриц.)")


def split_and_pick(text: str, delimiter: str, index: int = 0) -> str:
    """
    Делит строку и возвращает элемент по индексу.
    Поддерживает отрицательные индексы à-la Python.
    """
    parts: List[str] = text.split(delimiter)
    try:
        return parts[index]
    except IndexError:
        return ""


split_pick_tool = StructuredTool.from_function(
    func=split_and_pick,
    name="split_and_pick",
    description="text.split(delimiter)[index] c защитой от IndexError.",
    args_schema=SplitPickArgs,
)
if CrewStructuredTool:
    crew_split_pick_tool = CrewStructuredTool.from_function(
        split_and_pick,
        name="split_and_pick",
        description="text.split(delimiter)[index] c защитой от IndexError.",
        args_schema=SplitPickArgs,
    )
else:
    crew_split_pick_tool = None

# ──────────────────────── 5. Regex extract ────────────────────────────
import re  # noqa: E402


class RegexArgs(BaseModel):
    text: str = Field(..., description="Оригинальный текст")
    pattern: str = Field(..., description="Регулярка (Python re)")
    group: int = Field(
        0,
        description="Какую группу вернуть. 0 — всё совпадение. "
        "Если группы нет/нет совпадений — вернуть ''.",
    )


def regex_extract(text: str, pattern: str, group: int = 0) -> str:
    """Возвращает первую группу (или полное совпадение) по regex."""
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return ""
    try:
        return match.group(group)
    except IndexError:
        return ""


regex_tool = StructuredTool.from_function(
    func=regex_extract,
    name="regex_extract",
    description="Извлечь фрагмент по регулярке. Поддерживает группы.",
    args_schema=RegexArgs,
    # handle_tool_error=True  # если хотите возвращать текст ошибки вместо exception
)
if CrewStructuredTool:
    crew_regex_tool = CrewStructuredTool.from_function(
        regex_extract,
        name="regex_extract",
        description="Извлечь фрагмент по регулярке. Поддерживает группы.",
        args_schema=RegexArgs,
    )
else:
    crew_regex_tool = None

# ──────────────────────── Registry ────────────────────────────────────
STRING_TOOLS = [
    before_tool,
    after_tool,
    between_tool,
    split_pick_tool,
    regex_tool,
]
if CrewStructuredTool:
    CREW_STRING_TOOLS = [
        crew_before_tool,
        crew_after_tool,
        crew_between_tool,
        crew_split_pick_tool,
        crew_regex_tool,
    ]
else:
    CREW_STRING_TOOLS = []

# Convenience export: agent-friendly flat list
__all__: List[str] = ["STRING_TOOLS", "CREW_STRING_TOOLS"]


