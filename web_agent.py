from typing import Dict, Any, Optional, List
import json
from base_agent import BaseAgent, ReActStep
from orchestrator import Task, FactStore
from googleapiclient.discovery import build
import httpx
from bs4 import BeautifulSoup
import asyncio
import os

class WebAgent(BaseAgent):
    def __init__(self, agent_id: str, system_prompt: str, google_api_key: str, search_engine_id: str):
        super().__init__(agent_id, system_prompt)
        self.google_api_key = google_api_key
        self.search_engine_id = search_engine_id
        self.register_action("google_search", self.google_search)
        self.register_action("extract_content", self.extract_content)
        self.register_action("summarize_content", self.summarize_content)
        
    async def google_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Perform a Google search and return results"""
        print(f"Searching Google for: {query}")
        service = build("customsearch", "v1", developerKey=self.google_api_key)
        
        try:
            result = service.cse().list(
                q=query,
                cx=self.search_engine_id,
                num=num_results
            ).execute()
            
            return {
                "items": [
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": item.get("link", "")
                    }
                    for item in result.get("items", [])
                ]
            }
        except Exception as e:
            print(f"Google search error: {str(e)}")
            return {"items": []}
            
    async def extract_content(self, url: str) -> str:
        """Extract main content from a webpage"""
        print(f"Extracting content from: {url}")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                    
                # Get text content
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text[:2000]  # Limit length to avoid token limits
        except Exception as e:
            print(f"Content extraction error: {str(e)}")
            return ""
            
    async def summarize_content(self, text: str) -> str:
        """Summarize the extracted content using LLM"""
        print("Summarizing content...")
        try:
            import litellm
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the following text concisely:"
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=150
            )
            return response.choices[0].message["content"]
        except Exception as e:
            print(f"Summarization error: {str(e)}")
            return ""
            
    async def think(self, task: Task, fact_store: FactStore) -> ReActStep:
        """Process the task and decide next action"""
        # Get current facts
        facts = fact_store.facts
        
        # If we haven't searched yet, start with a search
        if "search_results" not in facts:
            return ReActStep(
                thought="We need to search for information first",
                action="google_search",
                action_input={"query": task.description, "num_results": 5}
            )
            
        # If we have search results but haven't processed them
        search_results = facts.get("search_results", {}).get("items", [])
        processed_urls = facts.get("processed_urls", set())
        
        for result in search_results:
            url = result.get("link")
            if url and url not in processed_urls:
                return ReActStep(
                    thought=f"Let's extract content from {url}",
                    action="extract_content",
                    action_input={"url": url}
                )
                
        # If we've processed all URLs, we're done
        return ReActStep(
            thought="We have collected all available information",
            action=None
        )
        
    async def observe(self, action_result: Any) -> str:
        """Process the result and update facts"""
        if isinstance(action_result, dict) and "items" in action_result:
            # This is a search result
            return f"Found {len(action_result['items'])} search results"
        elif isinstance(action_result, str):
            if len(action_result) > 100:
                # This is extracted content
                summary = await self.summarize_content(action_result)
                return f"Extracted and summarized content: {summary}"
            return action_result
        return str(action_result) 