import os
import pathlib
from .base import BaseTool
from ..core.llm import LLMClient
from ..core.prompts import TOOL_DESC_SEARCH_WEB

class SearchWebTool(BaseTool):
    """A tool to search the web for up-to-date information."""
    def __init__(self, llm_client: LLMClient):
        super().__init__(
            name="search_web",
            description=TOOL_DESC_SEARCH_WEB,
            underlying_model="Tavily"
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

    def execute(self, query: str) -> str:
        if not query:
            return "Error: query parameter is required."
            
        if not self.tavily_client:
            return "Error: Tavily API key not found in ~/tavily_api_key.txt or tavily-python is not installed. The search_web tool is not available."
        
        try:
            search_results = self.tavily_client.search(
                query=query, search_depth="advanced", max_results=5, include_answer=True)

            context = ""
            sources = []
            if 'results' in search_results:
                for result in search_results['results']:
                    url = result.get('url')
                    title = (result.get('title') or '').strip()
                    context += f"URL: {url}\nContent: {result.get('content')}\n---\n"
                    if url:
                        sources.append(f"- {url}" + (f" — {title}" if title else ""))

            if not context:
                return f"No search results found for the query: '{query}'"

            # Fixed: use correct parameter names (text, query) not (text_to_summarize, query)
            summary = self.llm_client.summarize_text(text=context, query=query)

            # Preserve provenance: the summary loses the URLs, but the agent often
            # needs to cite a source or open the most relevant page with the
            # browser tool, so list the sources alongside the summary.
            if sources:
                summary = f"{summary}\n\nSOURCES:\n" + "\n".join(sources)

            # Surface a direct answer from Tavily when present (often the most
            # precise response to a factual query).
            answer = search_results.get('answer') if isinstance(search_results, dict) else None
            if answer:
                summary = f"DIRECT ANSWER: {answer}\n\n{summary}"

            return summary

        except Exception as e:
            return f"An error occurred during the web search: {type(e).__name__}: {e}"
