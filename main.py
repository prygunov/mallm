from __future__ import annotations
import asyncio

from agents.coordinator_agent import run as run_coordinator

# Example query. Replace or pass via CLI as needed.
QUERY = (
    "What is the last word before the second chorus of the King of Pop's fifth single from his sixth studio album?"
)


def main(query: str = QUERY) -> None:
    result = asyncio.run(run_coordinator(query))
    print(result)


if __name__ == "__main__":
    main()
