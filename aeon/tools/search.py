import os
import subprocess
import requests
from .base import BaseTool
from ..core.llm import LLMClient
from ..core.prompts import TOOL_DESC_SEARCH_WEB

SEARXNG_PORT = os.environ.get("AEON_SEARXNG_PORT", "8095")
SEARXNG_URL = f"http://localhost:{SEARXNG_PORT}"


class SearchWebTool(BaseTool):
    """Quick web lookups via a LOCAL SearXNG metasearch container.

    Aeon is local-only: this hits an on-machine SearXNG instance that aggregates
    public search engines, so no third-party search API/SaaS (e.g. Tavily) sees or
    filters the queries. SafeSearch is forced off (uncensored). Best for shallow,
    public, easily-found information; deep digging into a specific site or dataset
    is the browser tools' job.
    """
    def __init__(self, llm_client: LLMClient):
        super().__init__(
            name="search_web",
            description=TOOL_DESC_SEARCH_WEB,
            underlying_model="SearXNG (local metasearch)",
        )
        self.llm_client = llm_client

    def _ensure_searxng(self):
        """Start the local SearXNG container if it isn't already healthy."""
        try:
            if requests.get(f"{SEARXNG_URL}/healthz", timeout=2).status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "start_searxng.sh"))
        try:
            subprocess.run(["bash", script], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to start local SearXNG: {(e.stderr or e.stdout or '').strip()}")

    @staticmethod
    def _instant_answers(data: dict) -> str:
        """SearXNG may return engine 'answers' (instant answers / infoboxes).
        Normalize the list (strings in some versions, dicts in others) to text."""
        out = []
        for a in data.get("answers") or []:
            if isinstance(a, dict):
                out.append((a.get("answer") or a.get("content") or "").strip())
            elif a:
                out.append(str(a).strip())
        return " ".join(t for t in out if t)

    def execute(self, query: str, max_results: int = 5) -> str:
        if not query:
            return "Error: query parameter is required."

        try:
            self._ensure_searxng()
            resp = requests.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json", "safesearch": 0,
                        "categories": "general", "language": "en"},
                timeout=30,
            )
            if resp.status_code != 200:
                return (f"Error: local SearXNG returned HTTP {resp.status_code}. "
                        f"{resp.text[:300]}")
            data = resp.json()
        except Exception as e:
            return f"An error occurred during the web search: {type(e).__name__}: {e}"

        results = (data.get("results") or [])[:max_results]
        if not results:
            return f"No search results found for the query: '{query}'"

        context = ""
        sources = []
        for r in results:
            url = r.get("url")
            title = (r.get("title") or "").strip()
            snippet = (r.get("content") or "").strip()
            context += f"URL: {url}\nTitle: {title}\nContent: {snippet}\n---\n"
            if url:
                sources.append(f"- {url}" + (f" — {title}" if title else ""))

        summary = self.llm_client.summarize_text(text=context, query=query)
        if sources:
            summary = f"{summary}\n\nSOURCES:\n" + "\n".join(sources)

        answer = self._instant_answers(data)
        if answer:
            summary = f"DIRECT ANSWER: {answer}\n\n{summary}"

        return summary
