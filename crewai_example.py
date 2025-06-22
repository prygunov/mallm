from __future__ import annotations
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process

from tools.open_url import open_url_tool
from tools.google_search import google_search_tool
from tools.browser_use import browser_tool
from tools.calculate_tool import calculate_tool

load_dotenv()

llm_main = ChatOpenAI(model="gpt-4o")
llm_planner = ChatOpenAI(model="gpt-4o-mini")

planner = Agent(
    role="Planner",
    goal="Break down the user query into a simple plan",
    backstory="Helps other agents by providing short numbered steps",
    allow_delegation=False,
    llm=llm_planner,
)

researcher = Agent(
    role="Researcher",
    goal="Use the available tools to complete the plan and answer the query",
    backstory="Expert web researcher able to browse pages and perform calculations",
    tools=[google_search_tool, open_url_tool, browser_tool, calculate_tool],
    allow_delegation=False,
    llm=llm_main,
)

task_plan = Task(
    description="Create a numbered plan for solving the query: {query}",
    expected_output="A list of concise steps",
    agent=planner,
)

task_execute = Task(
    description="Follow the plan below to answer: {query}\nPlan:\n{plan}",
    expected_output="Final answer to the user question",
    agent=researcher,
)

crew = Crew(
    agents=[planner, researcher],
    tasks=[task_plan, task_execute],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    query = (
        "What is the last word before the second chorus of the King of Pop's fifth "
        "single from his sixth studio album?"
    )
    result = crew.kickoff(inputs={"query": query})
    print("\nFinal answer:\n", result.final_output)
