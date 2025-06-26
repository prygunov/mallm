from __future__ import annotations
import asyncio

from agents.coordinator_agent import run as run_coordinator
from agents.critic_agent import run_critic

# Example query. Replace or pass via CLI as needed.
QUERY = (
    "What is the last word before the second chorus of the King of Pop's fifth single from his sixth studio album?"
)


def main(query: str = QUERY) -> None:
    coord_result = asyncio.run(run_coordinator(query))
    final_result = asyncio.run(run_critic(coord_result))
    print(final_result)


if __name__ == "__main__":
    main()
