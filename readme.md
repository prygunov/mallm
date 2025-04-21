# Multi-Agent LLM System with Web Research Capabilities

This project implements a multi-agent system using large language models (LLMs) with web research capabilities. The system consists of a coordinator agent that manages tasks and a web agent that performs internet research.

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_google_search_engine_id
```

### Getting API Keys

- **OpenAI API Key**: Sign up at [OpenAI](https://platform.openai.com) and create an API key
- **Google API Key**: 
  1. Go to [Google Cloud Console](https://console.cloud.google.com)
  2. Create a new project or select an existing one
  3. Enable the Custom Search API
  4. Create credentials (API key)
- **Google Search Engine ID**:
  1. Go to [Google Programmable Search Engine](https://programmablesearchengine.google.com)
  2. Create a new search engine
  3. Get the Search Engine ID from the setup page

## Usage

Run the main script:
```bash
python main.py
```

The system will:
1. Initialize the coordinator and web agent
2. Process the input task
3. Delegate research tasks to the web agent
4. Coordinate task execution
5. Present the results and gathered information

## Features

- Multi-agent coordination
- Web research capabilities using Google Custom Search
- Flexible task delegation based on requirements
- Structured information gathering and processing
- Environment variable management for secure API key handling

## Project Structure

- `main.py`: Main script that initializes and coordinates the agents
- `orchestrator.py`: Manages the coordination between different agents
- `base_agent.py`: Contains the base agent class with common functionality
- `web_agent.py`: Implementation of the web research agent
- `llm_agent.py`: Implementation of the LLM-based agent for processing tasks
- `task_planner.py`: Handles task planning and decomposition
- `react.py`: Implements the ReAct (Reasoning and Acting) framework
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not tracked in git)

## Jupyter Notebooks

The project includes two Jupyter notebooks for development and testing:

- `MALLM.ipynb`: Main development notebook containing experiments and demonstrations of the multi-agent system
- `react.ipynb`: Notebook focused on testing and demonstrating the ReAct framework implementation

These notebooks are useful for interactive development and understanding the system's behavior.

## Search Loop Implementation

The system implements a ReAct (Reasoning and Acting) loop for web searches:

1. **Initial Search**: When a task begins, the WebAgent performs a Google search using the task description
2. **URL Processing Loop**:
   - For each search result URL:
     1. Extract content from the webpage
     2. Summarize the extracted content using LLM
     3. Store the processed URL to avoid duplicates
   - Continue until all URLs are processed
3. **Fact Storage**: All gathered information is stored in a central fact store for use by other agents

The loop ensures thorough information gathering while avoiding duplicate processing of URLs.
