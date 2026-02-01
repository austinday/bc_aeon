import os
import pathlib
from .base import BaseTool
from ..core.llm import LLMClient

class SearchWebTool(BaseTool):
    """A tool to search the web for up-to-date information."""
    def __init__(self, llm_client: LLMClient):
        super().__init__(
            name="search_web",
            description="""Web search. Use for external info/docs, current information, or inspiration when stuck. Params: `query` (str). Example: `{"tool_name": "search_web", "parameters": {"query": "python singleton pattern"}}`"""
        )
        self.llm_client = llm_client
        self.tavily_client = None
        
        try:
            from tavily import TavilyClient
            api_key_path = pathlib.Path.home() / "tavily_api_key.txt"
            api_key = None
            if api_key_path.is_file():
                with open(api_key_path, 'r') as f:
                    api_key = f.readline().strip()

            if api_key:
                self.tavily_client = TavilyClient(api_key=api_key)
        except ImportError:
            pass # tavily_client remains None

    def execute(self, query: str):
        if not self.tavily_client:
            return "Error: Tavily API key not found in ~/tavily_api_key.txt or tavily-python is not installed. The search_web tool is not available."
        
        try:
            search_results = self.tavily_client.search(query=query, search_depth="advanced", max_results=5)
            
            context = ""
            if 'results' in search_results:
                for result in search_results['results']:
                    context += f"URL: {result.get('url')}\nContent: {result.get('content')}\n---\n"
            
            if not context:
                return f"No search results found for the query: '{query}'"

            summary = self.llm_client.summarize_text(text_to_summarize=context, query=query)
            return summary

        except Exception as e:
            return f"An error occurred during the web search: {e}"