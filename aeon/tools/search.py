import os
import json
from pathlib import Path
import re
import stat
import requests
from .base import BaseTool
from ..core.llm import LLMClient
from ..core.prompts import TOOL_DESC_SEARCH_WEB


_SERVICE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_INSTANCE_PREFIX = "Aeon SearXNG "
_SEARXNG_IMAGE_ID = (
    "sha256:892cf809341915a4b7710d3c9045005b4c377d51335a089b6d4da0b28750788d"
)

def _searxng_loopback_url() -> str:
    raw = os.environ.get("AEON_SEARXNG_PORT", "8095").strip()
    if not raw.isascii() or not raw.isdecimal():
        raise RuntimeError("AEON_SEARXNG_PORT must be a decimal loopback port")
    port = int(raw)
    if not 1 <= port <= 65535:
        raise RuntimeError("AEON_SEARXNG_PORT is outside the valid port range")
    return f"http://127.0.0.1:{port}"


def _service_identity() -> str:
    receipt = Path(
        os.environ.get(
            "AEON_SEARXNG_RECEIPT",
            "~/.aeon/host-services/searxng/service.json",
        )
    ).expanduser()
    metadata = receipt.lstat()
    parent = receipt.parent.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
        or not 0 < metadata.st_size <= 64 * 1024
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or parent.st_mode & 0o077
    ):
        raise RuntimeError("the local SearXNG ownership receipt is unsafe")
    document = json.loads(receipt.read_text(encoding="utf-8"))
    service_id = str(document.get("service_id", "")) if isinstance(document, dict) else ""
    if (
        not _SERVICE_ID_RE.fullmatch(service_id)
        or document.get("schema") != 1
        or document.get("image_id") != _SEARXNG_IMAGE_ID
    ):
        raise RuntimeError("the local SearXNG ownership receipt is invalid")
    return service_id


def _local_get(path: str, *, timeout: int, params: dict | None = None):
    if not path.startswith("/") or "//" in path or "?" in path or "#" in path:
        raise RuntimeError("invalid local SearXNG request path")
    with requests.Session() as session:
        session.trust_env = False
        return session.get(
            f"{_searxng_loopback_url()}{path}",
            params=params,
            timeout=timeout,
            allow_redirects=False,
        )


class SearchWebTool(BaseTool):
    """Quick web lookups via the operator-managed local SearXNG service.

    This hits an on-machine SearXNG instance that aggregates
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
        """Require the reviewed local service without controlling containers."""
        try:
            service_id = _service_identity()
            health = _local_get("/healthz", timeout=2)
            if (
                health.status_code == 200
                and health.content == b"OK"
                and health.headers.get("content-type", "").startswith("text/plain")
            ):
                config = _local_get("/config", timeout=2)
                if len(config.content) > 512 * 1024:
                    raise RuntimeError("local SearXNG identity response is oversized")
                document = config.json() if config.status_code == 200 else {}
                if (
                    isinstance(document, dict)
                    and document.get("instance_name")
                    == f"{_INSTANCE_PREFIX}{service_id}"
                    and isinstance(document.get("version"), str)
                    and isinstance(document.get("engines"), list)
                ):
                    return
        except Exception:
            pass
        raise RuntimeError(
            "The operator-managed CPU-only SearXNG service is unavailable. "
            "search_web does not inspect, start, replace, or stop host containers."
        )

    @staticmethod
    def _instant_answers(data: dict) -> str:
        """SearXNG may return engine 'answers' (instant answers / infoboxes).
        Normalize the list (strings in some versions, dicts in others) to text."""
        out = []
        for a in data.get("answers") or []:
            if isinstance(a, dict):
                out.append(str(a.get("answer") or a.get("content") or "").strip())
            elif a:
                out.append(str(a).strip())
        return " ".join(t for t in out if t)

    def execute(self, query: str, max_results: int = 5) -> str:
        query = str(query or "").strip()
        if not query:
            return "Error: query parameter is required."
        if len(query) > 2000:
            return "Error: query exceeds the 2,000-character limit."
        try:
            max_results = max(1, min(int(max_results), 10))
        except (TypeError, ValueError):
            return "Error: max_results must be an integer from 1 to 10."

        try:
            self._ensure_searxng()
            resp = _local_get(
                "/search",
                params={"q": query, "format": "json", "safesearch": 0,
                        "categories": "general", "language": "en"},
                timeout=30,
            )
            if resp.status_code != 200:
                return (f"Error: local SearXNG returned HTTP {resp.status_code}. "
                        f"{resp.text[:300]}")
            if len(resp.content) > 2 * 1024 * 1024:
                return "Error: local SearXNG response exceeded the 2 MiB limit."
            data = resp.json()
            if not isinstance(data, dict):
                return "Error: local SearXNG returned an invalid response."
        except Exception as e:
            return f"An error occurred during the web search: {type(e).__name__}: {e}"

        raw_results = data.get("results") or []
        if not isinstance(raw_results, list):
            return "Error: local SearXNG returned an invalid results list."
        results = [item for item in raw_results if isinstance(item, dict)][:max_results]
        if not results:
            return f"No search results found for the query: '{query}'"

        context = ""
        sources = []
        for r in results:
            url = str(r.get("url") or "").strip()
            title = str(r.get("title") or "").strip()
            snippet = str(r.get("content") or "").strip()
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
