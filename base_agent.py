from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import json
from dataclasses import dataclass
from orchestrator import Task, FactStore

@dataclass
class ReActStep:
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation
        }

class BaseAgent(ABC):
    def __init__(self, agent_id: str, system_prompt: str):
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, Any]] = []
        self.available_actions: Dict[str, Any] = {}
        
    def register_action(self, action_name: str, action_func: Any):
        """Register an action that the agent can use"""
        self.available_actions[action_name] = action_func
        
    @abstractmethod
    async def think(self, task: Task, fact_store: FactStore) -> ReActStep:
        """Generate the next thought and action based on the task and facts"""
        pass
        
    @abstractmethod
    async def observe(self, action_result: Any) -> str:
        """Process the result of an action and generate an observation"""
        pass
        
    async def execute_action(self, action: str, action_input: Dict[str, Any]) -> Any:
        """Execute a registered action"""
        if action not in self.available_actions:
            raise ValueError(f"Unknown action: {action}")
        action_func = self.available_actions[action]
        return await action_func(**action_input)
        
    async def execute(self, task: Task, fact_store: FactStore) -> Any:
        """Execute the ReAct loop for a given task"""
        steps: List[ReActStep] = []
        
        while True:
            # Think
            step = await self.think(task, fact_store)
            
            # If no action is proposed, we're done
            if not step.action:
                break
                
            # Execute action
            try:
                action_result = await self.execute_action(step.action, step.action_input or {})
                # Observe
                step.observation = await self.observe(action_result)
            except Exception as e:
                step.observation = f"Error: {str(e)}"
                
            steps.append(step)
            
        # Return the final result
        return {
            "steps": [step.to_dict() for step in steps],
            "final_thought": step.thought
        }
        
    def _format_conversation_history(self) -> str:
        """Format the conversation history for the LLM prompt"""
        formatted = []
        for msg in self.conversation_history:
            if msg["role"] == "system":
                formatted.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                formatted.append(f"Human: {msg['content']}")
            elif msg["role"] == "assistant":
                formatted.append(f"Assistant: {msg['content']}")
        return "\n".join(formatted) 