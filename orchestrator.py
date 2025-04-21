from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import deque
import json

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    description: str
    priority: int
    dependencies: List[str]
    status: TaskStatus
    result: Optional[Any] = None
    assigned_agent: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result,
            "assigned_agent": self.assigned_agent
        }

class FactStore:
    def __init__(self):
        self.facts: Dict[str, Any] = {}
        
    def add_fact(self, key: str, value: Any):
        self.facts[key] = value
        print(f"Added fact: {key} = {value}")
        
    def get_fact(self, key: str) -> Optional[Any]:
        return self.facts.get(key)
        
    def to_json(self) -> str:
        return json.dumps(self.facts)

class Orchestrator:
    def __init__(self):
        self.tasks: deque[Task] = deque()
        self.agents: Dict[str, Any] = {}
        self.fact_store = FactStore()
        self.completed_tasks: List[Task] = []
        
    def register_agent(self, agent_id: str, agent: Any):
        """Register a new agent with the orchestrator"""
        print(f"Registering agent: {agent_id}")
        self.agents[agent_id] = agent
        
    def add_task(self, task: Task):
        """Add a new task to the task queue"""
        print(f"Adding task: {task.id} - {task.description}")
        self.tasks.append(task)
        
    def get_next_task(self) -> Optional[Task]:
        """Get the next available task that has all dependencies met"""
        if not self.tasks:
            return None
            
        # First, sort tasks by priority (lower number = higher priority)
        task_list = list(self.tasks)
        task_list.sort(key=lambda x: x.priority)
        self.tasks = deque(task_list)
            
        for i in range(len(self.tasks)):
            task = self.tasks[0]
            # Check if all dependencies are completed
            deps_met = all(
                any(t.id == dep and t.status == TaskStatus.COMPLETED 
                    for t in self.completed_tasks)
                for dep in task.dependencies
            )
            
            if deps_met:
                return self.tasks.popleft()
            self.tasks.rotate(-1)
            
        print("No tasks ready for execution (dependency constraints)")
        return None
        
    async def execute_task(self, task: Task):
        """Execute a task using the appropriate agent"""
        if not task.assigned_agent:
            print(f"Task {task.id} has no assigned agent")
            task.status = TaskStatus.FAILED
            task.result = "No agent assigned"
            return
            
        if task.assigned_agent not in self.agents:
            print(f"Agent {task.assigned_agent} not found for task {task.id}")
            task.status = TaskStatus.FAILED
            task.result = f"Agent {task.assigned_agent} not found"
            return
            
        agent = self.agents[task.assigned_agent]
        print(f"\nExecuting task: {task.id} - {task.description}")
        print(f"Using agent: {task.assigned_agent}")
        
        try:
            task.status = TaskStatus.IN_PROGRESS
            task.result = await agent.execute(task, self.fact_store)
            task.status = TaskStatus.COMPLETED
            self.completed_tasks.append(task)
            print(f"Task completed: {task.id}")
            print(f"Result: {task.result}")
            
            # If the task produced any results, store them as facts
            if isinstance(task.result, dict):
                for key, value in task.result.items():
                    self.fact_store.add_fact(f"{task.id}_{key}", value)
                    
        except Exception as e:
            print(f"Error executing task {task.id}: {str(e)}")
            task.status = TaskStatus.FAILED
            task.result = str(e)
        
    async def run(self):
        """Main execution loop"""
        print("\nStarting task execution...")
        while self.tasks:
            task = self.get_next_task()
            if task:
                await self.execute_task(task)
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
            else:
                # If no task is ready, wait a bit longer
                await asyncio.sleep(1)
                
        print("\nTask execution completed")
        print(f"Completed tasks: {len(self.completed_tasks)}")
        print(f"Known facts: {self.fact_store.to_json()}") 