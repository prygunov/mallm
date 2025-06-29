from __future__ import annotations
import asyncio

from agents.coordinator_agent import run as run_coordinator
from agents.critic_agent import run_critic

# Example query. Replace or pass via CLI as needed.
QUERY = (
    """.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI"""
)


def main(query: str = QUERY) -> None:
    coord_result = asyncio.run(run_coordinator(query))
    final_result = asyncio.run(run_critic(coord_result))
    if final_result != 'APPROVED':
        print(f"Critique: {final_result}")
        main(query)
    print(coord_result)


if __name__ == "__main__":
    main()
