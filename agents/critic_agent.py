from __future__ import annotations
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    critic_llm = ChatOpenAI(model="gpt-4o-mini")
else:
    critic_llm = None

CRITIC_SYSTEM_PROMPT = (
    "You are a critical reviewer. Evaluate the assistant's answer provided in the 'Answer' section. "
    "If the answer is clear, correct and safe, respond with only the word 'APPROVED'. "
    "Otherwise, respond with a short critique explaining the problem."
)

async def run_critic(answer: str) -> str:
    """Review the coordinator's answer and either approve or return a critique."""
    if not critic_llm:
        # If no LLM is configured, pass the answer through.
        return answer
    message = CRITIC_SYSTEM_PROMPT + "\n\nAnswer:\n" + answer
    result = await critic_llm.ainvoke(message)
    verdict = result.content.strip()
    return verdict.upper()
