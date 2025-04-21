from typing import List, Dict, Any
import litellm
import json
import uuid
from orchestrator import Task, TaskStatus

PLANNING_PROMPT = """You are a task planning AI. Your job is to break down complex tasks into smaller, manageable subtasks.
Each subtask should be atomic and achievable by a single agent.

Consider the following when breaking down tasks:
1. Dependencies between subtasks
2. Parallel execution opportunities
3. Information gathering needs
4. Verification and validation steps

You must respond with a JSON object containing a "subtasks" array. Each subtask must have:
- id: a unique string identifier
- description: what needs to be done
- priority: number 1-5 (1 is highest)
- dependencies: array of other task IDs this depends on
- estimated_duration: estimated time in minutes

Example response format:
{{
    "subtasks": [
        {{
            "id": "task1",
            "description": "Search for the song title and album",
            "priority": 1,
            "dependencies": [],
            "estimated_duration": "5"
        }}
    ]
}}

Task to break down: {task_description}

Known facts: {facts_json}

Remember to respond with valid JSON only, no additional text."""

class TaskPlanner:
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        
    async def plan_task(self, task_description: str, facts: Dict[str, Any] = None) -> List[Task]:
        """Break down a complex task into subtasks"""
        
        # Get planning from LLM
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a task planning AI. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": PLANNING_PROMPT.format(
                            task_description=task_description,
                            facts_json=json.dumps(facts or {})
                        )
                    }
                ],
                temperature=0
            )
            
            # Get the response content
            response_text = response.choices[0].message["content"].strip()
            
            # Parse the response
            plan = json.loads(response_text)
            
            if "subtasks" not in plan:
                raise ValueError("LLM response missing 'subtasks' key")
            
            # Convert to Task objects
            tasks = []
            for subtask in plan["subtasks"]:
                task = Task(
                    id=subtask.get("id", self.generate_task_id()),
                    description=subtask["description"],
                    priority=subtask["priority"],
                    dependencies=subtask.get("dependencies", []),
                    status=TaskStatus.PENDING
                )
                tasks.append(task)
                
            return tasks
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}\nResponse was: {response_text}")
        except KeyError as e:
            raise ValueError(f"Missing required field in LLM response: {e}\nResponse was: {response_text}")
        except Exception as e:
            raise ValueError(f"Error during task planning: {str(e)}")
            
    def generate_task_id(self) -> str:
        """Generate a unique task ID"""
        return str(uuid.uuid4()) 