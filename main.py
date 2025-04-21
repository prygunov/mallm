import asyncio
import os
from dotenv import load_dotenv
from orchestrator import Orchestrator
from task_planner import TaskPlanner
from llm_agent import LLMAgent
from web_agent import WebAgent
from browser_use import Browser, BrowserConfig, Controller

# Load environment variables
load_dotenv()

# Check for required environment variables
required_env_vars = [
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_SEARCH_ENGINE_ID"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Configure litellm
import litellm
litellm.api_key = os.getenv("OPENAI_API_KEY")
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Example system prompts
COORDINATOR_PROMPT = """You are a task coordinator AI. Your role is to manage and execute tasks while maintaining a clear chain of reasoning.
Follow the ReAct pattern:
1. Think about what needs to be done
2. Choose an appropriate action
3. Observe the results
4. Update your understanding

Available actions:
- search_web: Search the internet for information
- browse_url: Navigate to and interact with a webpage
- extract_info: Extract specific information from a webpage
- ask_human: Ask the user for clarification or information

For each task, follow the ReAct pattern and use the most appropriate action."""

WEB_AGENT_PROMPT = """You are a web research AI. Your role is to gather and analyze information from the internet.
Follow these steps:
1. Search for relevant information using Google
2. Extract content from web pages
3. Summarize and analyze the information
4. Store important facts for later use

Make sure to:
- Verify information from multiple sources
- Extract key facts and data points
- Summarize findings clearly
- Update the fact store with new information"""

async def search_web(query: str) -> str:
    """Simulate web search"""
    return f"Search results for: {query}"

async def browse_url(url: str) -> str:
    """Navigate to a URL"""
    return f"Browsed to: {url}"

async def extract_info(selector: str) -> str:
    """Extract information from current page"""
    return f"Extracted: {selector}"

async def ask_human(question: str) -> str:
    """Ask user for input"""
    print(f"\n{question}")
    return input("Your answer: ")

async def main():
    # Initialize components
    orchestrator = Orchestrator()
    task_planner = TaskPlanner(model="gpt-3.5-turbo")
    
    # Initialize agents
    coordinator = LLMAgent("coordinator", COORDINATOR_PROMPT, model="gpt-3.5-turbo")
    web_agent = WebAgent(
        "web_agent",
        WEB_AGENT_PROMPT,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        search_engine_id=os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    )
    
    # Register actions with coordinator
    coordinator.register_action("search_web", search_web)
    coordinator.register_action("browse_url", browse_url)
    coordinator.register_action("extract_info", extract_info)
    coordinator.register_action("ask_human", ask_human)
    
    # Register agents with orchestrator
    orchestrator.register_agent("coordinator", coordinator)
    orchestrator.register_agent("web_agent", web_agent)
    
    # Example task that requires web research
    main_task = "What is the last word before the second chorus of the King of Pop's fifth single from his sixth studio album?"
    
    try:
        # Plan the task
        print("Planning task...")
        subtasks = await task_planner.plan_task(main_task)
        
        print(f"\nCreated {len(subtasks)} subtasks:")
        for task in subtasks:
            print(f"- {task.description} (Priority: {task.priority})")
            # Assign research tasks to web_agent, others to coordinator
            if any(word in task.description.lower() for word in ["search", "research", "find", "gather", "collect"]):
                task.assigned_agent = "web_agent"
            else:
                task.assigned_agent = "coordinator"
            orchestrator.add_task(task)
        
        # Run the orchestrator
        print("\nExecuting tasks...")
        await orchestrator.run()
        
        # Print collected facts
        print("\nCollected facts:")
        for key, value in orchestrator.fact_store.facts.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Caused by: {str(e.__cause__)}")
    finally:
        # Clean up
        if 'browser' in locals():
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main()) 