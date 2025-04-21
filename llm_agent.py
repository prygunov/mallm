from typing import Dict, Any, Optional, List
import json
import litellm
from base_agent import BaseAgent, ReActStep
from orchestrator import Task, FactStore

class LLMAgent(BaseAgent):
    def __init__(self, agent_id: str, system_prompt: str, model: str = "gpt-4"):
        super().__init__(agent_id, system_prompt)
        self.model = model
        
    async def think(self, task: Task, fact_store: FactStore) -> ReActStep:
        # Construct the prompt
        prompt = self._construct_prompt(task, fact_store)
        
        # Get response from LLM
        print(f"\nThinking about task: {task.description}")
        response = await self._get_llm_response(prompt)
        print(f"LLM Response:\n{response}")
        
        # Parse the response into a ReActStep
        step = self._parse_react_response(response)
        print(f"Parsed step: {step.to_dict()}")
        return step
        
    async def observe(self, action_result: Any) -> str:
        """Process the action result and generate an observation"""
        if isinstance(action_result, (str, int, float, bool)):
            result = str(action_result)
        else:
            result = json.dumps(action_result)
        print(f"Observation: {result}")
        return result
        
    def _construct_prompt(self, task: Task, fact_store: FactStore) -> List[Dict[str, str]]:
        # Start with system prompt
        prompt = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history if any
        if self.conversation_history:
            prompt.extend(self.conversation_history)
        
        # Add task description
        task_desc = f"""
Current task: {task.description}

Available actions:
{self._format_available_actions()}

Known facts:
{fact_store.to_json()}

Previous steps:
{self._format_conversation_history()}

What should be done next? Use the following format:
Thought: (your reasoning)
Action: (action name or "FINISH" if done)
Action Input: (action parameters in JSON format)
        """.strip()
        
        prompt.append({"role": "user", "content": task_desc})
        return prompt
        
    def _format_available_actions(self) -> str:
        actions = []
        for action_name in self.available_actions:
            actions.append(f"- {action_name}")
        return "\n".join(actions)
        
    async def _get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """Get response from LLM"""
        try:
            print("Sending request to LLM...")
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=0
            )
            content = response.choices[0].message["content"]
            print("Received response from LLM")
            return content
        except Exception as e:
            print(f"Error in LLM call: {str(e)}")
            raise ValueError(f"Error getting LLM response: {str(e)}")
        
    def _parse_react_response(self, response: str) -> ReActStep:
        """Parse LLM response into a ReActStep"""
        lines = response.strip().split("\n")
        step = ReActStep(thought="")
        
        current_key = None
        current_value = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Thought:"):
                current_key = "thought"
                current_value = [line[8:].strip()]
            elif line.startswith("Action:"):
                if current_key == "thought":
                    step.thought = "\n".join(current_value)
                current_key = "action"
                action = line[7:].strip()
                if action != "FINISH":
                    step.action = action
                current_value = []
            elif line.startswith("Action Input:"):
                if current_key == "action":
                    step.action = "\n".join(current_value)
                current_key = "action_input"
                current_value = []
            elif current_key:
                current_value.append(line.strip())
        
        # Handle final key
        if current_key == "thought":
            step.thought = "\n".join(current_value)
        elif current_key == "action":
            step.action = "\n".join(current_value)
        elif current_key == "action_input":
            try:
                step.action_input = json.loads("\n".join(current_value))
            except json.JSONDecodeError as e:
                print(f"Error parsing action input JSON: {str(e)}")
                print(f"Raw input was: {current_value}")
                step.action_input = {}
                
        return step
        
    async def execute(self, task: Task, fact_store: FactStore) -> Any:
        """Execute the ReAct loop for a given task"""
        print(f"\nStarting execution of task: {task.description}")
        steps: List[ReActStep] = []
        max_steps = 10  # Prevent infinite loops
        step_count = 0
        
        while step_count < max_steps:
            step_count += 1
            print(f"\nStep {step_count}:")
            
            # Think
            step = await self.think(task, fact_store)
            
            # If no action is proposed, we're done
            if not step.action:
                print("No action proposed, finishing task")
                break
                
            # Execute action
            try:
                print(f"Executing action: {step.action}")
                action_result = await self.execute_action(step.action, step.action_input or {})
                # Observe
                step.observation = await self.observe(action_result)
            except Exception as e:
                print(f"Error executing action: {str(e)}")
                step.observation = f"Error: {str(e)}"
                
            steps.append(step)
            
            # Add step to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": f"Thought: {step.thought}\nAction: {step.action}\nAction Input: {json.dumps(step.action_input)}"
            })
            self.conversation_history.append({
                "role": "user",
                "content": f"Observation: {step.observation}"
            })
            
        # Return the final result
        result = {
            "steps": [step.to_dict() for step in steps],
            "final_thought": step.thought if steps else "No steps executed"
        }
        print(f"\nTask execution completed with {len(steps)} steps")
        return result 