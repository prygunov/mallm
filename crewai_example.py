from __future__ import annotations
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process

from tools.open_url import crew_open_url_tool
from tools.google_search import crew_google_search_tool
from tools.browser_use import crew_browser_tool
from tools.calculate_tool import crew_calculate_tool

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

researcher_tools = [
    crew_google_search_tool,
    crew_open_url_tool,
    crew_browser_tool,
    crew_calculate_tool,
]
researcher = Agent(
    role="Researcher",
    goal="Use the available tools to complete the plan and answer the query",
    backstory="Expert web researcher able to browse pages and perform calculations",
    tools=[tool for tool in researcher_tools if tool],
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

