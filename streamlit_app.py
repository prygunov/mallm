import asyncio
import streamlit as st

from main import MAX_STEPS, initial_plan, replan, agent_executor


async def run_query(query: str):
    plan_placeholder = st.empty()
    facts_placeholder = st.empty()
    thought_placeholder = st.empty()

    tasks = initial_plan(query)
    completed = []
    step = 0

    while tasks and step < MAX_STEPS:
        step += 1
        current_task = tasks.pop(0)
        plan_placeholder.markdown("**Текущий план:**\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate([current_task] + tasks)))
        facts_placeholder.markdown("**Текущие факты:**\n" + "\n".join(f"{i+1}. {t} - {r}" for i, (t, r) in enumerate(completed)))
        agent_input = f"Current task: {current_task}\nFacts: {'; '.join(f'{t} - {r}' for t, r in completed)}"
        result = await agent_executor.ainvoke({"input": agent_input})
        output = result["output"]
        thought_placeholder.markdown(f"**Шаг {step}:** {output}")
        completed.append((current_task, output))
        tasks = replan(query, completed)

    plan_placeholder.markdown("**План выполнен.**")
    facts_placeholder.markdown("**Факты:**\n" + "\n".join(f"{i+1}. {t} - {r}" for i, (t, r) in enumerate(completed)))


def main():
    st.title("Mallm Agent UI")
    query = st.text_input("Введите запрос")
    if st.button("Отправить") and query:
        asyncio.run(run_query(query))


if __name__ == "__main__":
    main()
