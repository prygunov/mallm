# Multi-Agent LLM Experiments

This repository contains resources for experimenting with large language models in a multi-agent environment. The main notebook demonstrates using browser automation together with an LLM to research information on the web.

## Contents

- `metadata.jsonl` – A JSON Lines file with question and task metadata used in experiments.
- `questions` – A text file with a couple of sample prompts.
- `react-agent.ipynb` – Jupyter notebook showcasing an agent that uses Playwright and OpenAI models to search the web and summarize results.
- `requirements.txt` – Python dependencies for running the notebook.

## Setup

1. Create and activate a Python virtual environment.
2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Install Playwright browser binaries (needed for web automation):

```bash
python -m playwright install
```

4. Create a `.env` file and provide your API keys, e.g.:

```
OPENAI_API_KEY=your_openai_key
```

Additional environment variables may be necessary depending on the search service you use. Review the notebook for details.

## Using the Notebook

Open `react-agent.ipynb` in Jupyter or VS Code and run the cells sequentially. The notebook demonstrates:

- Loading environment variables and configuring OpenAI models.
- Launching a headless browser with Playwright.
- Issuing search queries, scraping pages, and summarizing content with the LLM.
- An interactive loop that allows you to respond to agent questions while it performs research.

Feel free to modify the code or prompts to suit your own experiments.

## Working with the Dataset

`metadata.jsonl` consists of JSON objects, one per line. Each object contains a `task_id` and a `Question` field. You can load it in Python as follows:

```python
import json
with open("metadata.jsonl", "r") as f:
    records = [json.loads(line) for line in f]
```

Use this dataset to evaluate your agent or generate new prompts.

---

This project is a minimal proof of concept for multi-agent web research with LLMs. Contributions are welcome.
