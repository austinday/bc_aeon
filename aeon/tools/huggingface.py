"""Bounded, read-only access to Hugging Face's public model metadata.

These tools deliberately use fixed Hugging Face HTTPS endpoints without local
credentials, browser state, proxies, or an LLM summarization pass.  The result
is evidence from the Hub response, not a claim that a model card is correct or
that an empty search proves the absence of a competing repository.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

import requests

from aeon.core.prompts import (
    TOOL_DESC_HUGGINGFACE_MODEL_INFO,
    TOOL_DESC_HUGGINGFACE_MODEL_SEARCH,
    TOOL_DESC_HUGGINGFACE_REPO_FILE,
)
from aeon.tools.base import BaseTool


_HF_ORIGIN = "https://huggingface.co"
_MODEL_API_PATH = "/api/models"
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_FILE_BYTES = 1024 * 1024
_MAX_REDIRECTS = 3
_REPO_PART_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$")
_FILTER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,199}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._/-]{0,198}[A-Za-z0-9])?$")
_SORT_FIELDS = {
    "last_modified": "lastModified",
    "trending_score": "trendingScore",
    "created_at": "createdAt",
    "downloads": "downloads",
    "likes": "likes",
}
_SEARCH_FIELDS = (
    "id",
    "modelId",
    "author",
    "sha",
    "createdAt",
    "lastModified",
    "private",
    "gated",
    "disabled",
    "downloads",
    "downloadsAllTime",
    "likes",
    "trendingScore",
    "pipeline_tag",
    "library_name",
)
_CONFIG_FIELDS = (
    "architectures",
    "model_type",
    "torch_dtype",
    "quantization_config",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "num_attention_heads",
    "num_key_value_heads",
    "text_config",
    "vision_config",
    "auto_map",
)
_CARD_FIELDS = (
    "license",
    "license_name",
    "license_link",
    "base_model",
    "datasets",
    "language",
    "library_name",
    "pipeline_tag",
    "tags",
    "model_name",
    "inference",
    "new_version",
)


class HuggingFacePublicAPIError(RuntimeError):
    """The fixed public endpoint returned an unsafe or invalid response."""


def _valid_repo_id(repo_id: str) -> str:
    value = str(repo_id or "").strip()
    parts = value.split("/")
    if len(parts) not in {1, 2} or any(not _REPO_PART_RE.fullmatch(part) for part in parts):
        raise ValueError("repo_id must be a Hugging Face model id such as 'org/model'")
    return value


def _valid_repo_path(path: str) -> str:
    value = str(path or "").strip().lstrip("/")
    parts = value.split("/")
    if (
        not value
        or len(value) > 512
        or any(part in {"", ".", ".."} for part in parts)
        or any("\\" in part or "\x00" in part for part in parts)
    ):
        raise ValueError("path must be a relative repository file path without traversal")
    return value


def _same_origin_url(value: str, *, allowed_paths: tuple[str, ...]) -> str:
    parsed = urlsplit(str(value or ""))
    path_allowed = any(
        parsed.path == prefix.rstrip("/")
        or parsed.path.startswith(prefix.rstrip("/") + "/")
        for prefix in allowed_paths
    )
    if (
        parsed.scheme != "https"
        or parsed.hostname != "huggingface.co"
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not path_allowed
    ):
        raise HuggingFacePublicAPIError("Hugging Face returned an unsafe continuation URL")
    return parsed.geturl()


def _read_bounded(response: requests.Response, *, maximum: int) -> bytes:
    raw_length = response.headers.get("content-length", "").strip()
    if raw_length.isdecimal() and int(raw_length) > maximum:
        raise HuggingFacePublicAPIError("Hugging Face response exceeded the size limit")
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > maximum:
            raise HuggingFacePublicAPIError("Hugging Face response exceeded the size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _public_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    maximum: int,
    redirect_paths: tuple[str, ...],
) -> tuple[bytes, requests.Response]:
    current = _same_origin_url(url, allowed_paths=redirect_paths)
    current_params = params
    with requests.Session() as session:
        session.trust_env = False
        for redirect_number in range(_MAX_REDIRECTS + 1):
            response = session.get(
                current,
                params=current_params,
                headers={
                    "accept": "application/json, text/plain; q=0.9",
                    "user-agent": "Aeon-HuggingFace-Public-Metadata/1",
                },
                timeout=(5, 20),
                allow_redirects=False,
                stream=True,
            )
            current_params = None
            if response.status_code in {301, 302, 303, 307, 308}:
                if redirect_number >= _MAX_REDIRECTS:
                    raise HuggingFacePublicAPIError("Hugging Face returned too many redirects")
                location = response.headers.get("location", "")
                current = _same_origin_url(
                    urljoin(current, location),
                    allowed_paths=redirect_paths,
                )
                response.close()
                continue
            if response.status_code != 200:
                raise HuggingFacePublicAPIError(
                    f"Hugging Face returned HTTP {response.status_code}"
                )
            return _read_bounded(response, maximum=maximum), response
    raise HuggingFacePublicAPIError("Hugging Face request did not complete")


def _public_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    body, response = _public_get(
        url,
        params=params,
        maximum=_MAX_JSON_BYTES,
        redirect_paths=(_MODEL_API_PATH,),
    )
    try:
        document = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HuggingFacePublicAPIError("Hugging Face returned invalid JSON") from exc
    next_url = ""
    candidate = response.links.get("next", {}).get("url", "")
    if candidate:
        next_url = _same_origin_url(candidate, allowed_paths=(_MODEL_API_PATH,))
    return document, next_url


def _bounded_list(value: Any, maximum: int) -> tuple[list[Any], int]:
    items = value if isinstance(value, list) else []
    return items[:maximum], max(0, len(items) - maximum)


def _search_item(item: dict[str, Any]) -> dict[str, Any]:
    selected = {field: item[field] for field in _SEARCH_FIELDS if field in item}
    tags, omitted = _bounded_list(item.get("tags"), 100)
    selected["tags"] = tags
    if omitted:
        selected["omitted_tag_count"] = omitted
    return selected


def _selected_mapping(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {field: value[field] for field in fields if field in value}


def _error(exc: Exception) -> str:
    detail = str(exc).splitlines()[0][:300] if str(exc) else type(exc).__name__
    return f"Error: {type(exc).__name__}: {detail}"


class HuggingFaceModelSearchTool(BaseTool):
    """Search the fixed public Hub model metadata endpoint."""

    def __init__(self):
        super().__init__(
            name="huggingface_model_search",
            description=TOOL_DESC_HUGGINGFACE_MODEL_SEARCH,
            underlying_model="Hugging Face public model API",
        )

    def execute(
        self,
        query: str = "",
        filter_tag: str = "",
        author: str = "",
        pipeline_tag: str = "",
        sort: str = "last_modified",
        direction: str = "desc",
        limit: int = 20,
        next_page_url: str = "",
    ) -> str:
        try:
            query = str(query or "").strip()
            if len(query) > 200:
                raise ValueError("query exceeds 200 characters")
            for label, value in (("filter_tag", filter_tag), ("pipeline_tag", pipeline_tag)):
                if value and not _FILTER_RE.fullmatch(str(value).strip()):
                    raise ValueError(f"{label} contains unsupported characters")
            author = str(author or "").strip()
            if author and not _REPO_PART_RE.fullmatch(author):
                raise ValueError("author is not a valid Hub namespace")
            if sort not in _SORT_FIELDS:
                raise ValueError("sort must be last_modified, trending_score, created_at, downloads, or likes")
            if direction not in {"asc", "desc"}:
                raise ValueError("direction must be 'asc' or 'desc'")
            if isinstance(limit, bool) or not 1 <= int(limit) <= 50:
                raise ValueError("limit must be an integer from 1 to 50")
            limit = int(limit)

            if next_page_url:
                if query or filter_tag or author or pipeline_tag:
                    raise ValueError("next_page_url cannot be combined with new search filters")
                url = _same_origin_url(next_page_url, allowed_paths=(_MODEL_API_PATH,))
                params = None
            else:
                url = f"{_HF_ORIGIN}{_MODEL_API_PATH}"
                params = {
                    "sort": _SORT_FIELDS[sort],
                    "direction": -1 if direction == "desc" else 1,
                    "limit": limit,
                    "full": "true",
                    "cardData": "true",
                    "config": "true",
                }
                if query:
                    params["search"] = query
                if filter_tag:
                    params["filter"] = str(filter_tag).strip()
                if author:
                    params["author"] = author
                if pipeline_tag:
                    params["pipeline_tag"] = str(pipeline_tag).strip()

            document, continuation = _public_json(url, params=params)
            if not isinstance(document, list) or any(not isinstance(item, dict) for item in document):
                raise HuggingFacePublicAPIError("Hugging Face returned an invalid model list")
            result = {
                "source": f"{_HF_ORIGIN}{_MODEL_API_PATH}",
                "result_count": len(document),
                "results": [_search_item(item) for item in document],
                "next_page_url": continuation or None,
                "evidence_limits": [
                    "An empty or short result set is not evidence that no matching or competing repository exists.",
                    "Search metadata and model-card fields are repository claims; inspect the exact repo, files, license text, and upstream sources before validation.",
                    "Repeat with relevant name variants, tags, sorts, pages, and derivative formats before making a coverage claim.",
                ],
            }
            return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        except (ValueError, TypeError, requests.RequestException, HuggingFacePublicAPIError) as exc:
            return _error(exc)


class HuggingFaceModelInfoTool(BaseTool):
    """Read exact metadata for one public Hub model repository."""

    def __init__(self):
        super().__init__(
            name="huggingface_model_info",
            description=TOOL_DESC_HUGGINGFACE_MODEL_INFO,
            underlying_model="Hugging Face public model API",
        )

    def execute(self, repo_id: str) -> str:
        try:
            repo_id = _valid_repo_id(repo_id)
            encoded = quote(repo_id, safe="/")
            document, _ = _public_json(f"{_HF_ORIGIN}{_MODEL_API_PATH}/{encoded}")
            if not isinstance(document, dict):
                raise HuggingFacePublicAPIError("Hugging Face returned invalid model metadata")

            files, omitted_files = _bounded_list(document.get("siblings"), 1000)
            tags, omitted_tags = _bounded_list(document.get("tags"), 500)
            top_level = {field: document[field] for field in _SEARCH_FIELDS if field in document}
            top_level["tags"] = tags
            result = {
                "source": f"{_HF_ORIGIN}/{repo_id}",
                "metadata": top_level,
                "config": _selected_mapping(document.get("config"), _CONFIG_FIELDS),
                "card_data": _selected_mapping(document.get("cardData"), _CARD_FIELDS),
                "safetensors": document.get("safetensors"),
                "gguf": document.get("gguf"),
                "transformers_info": document.get("transformersInfo"),
                "used_storage": document.get("usedStorage"),
                "files": files,
                "omitted_file_count": omitted_files,
                "omitted_tag_count": omitted_tags,
                "available_top_level_fields": sorted(str(key) for key in document),
                "evidence_limits": [
                    "Repository metadata establishes what the Hub returned for this revision, not that every card claim is true.",
                    "A license tag is not a redistribution ruling; inspect the actual license and upstream terms with huggingface_repo_file.",
                    "Parameter totals and hardware feasibility require artifact/config inspection and a truthful toolchain-specific resource calculation.",
                ],
            }
            return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        except (ValueError, TypeError, requests.RequestException, HuggingFacePublicAPIError) as exc:
            return _error(exc)


class HuggingFaceRepoFileTool(BaseTool):
    """Fetch a bounded public text file from a specific Hub model revision."""

    def __init__(self):
        super().__init__(
            name="huggingface_repo_file",
            description=TOOL_DESC_HUGGINGFACE_REPO_FILE,
            underlying_model="Hugging Face public repository file endpoint",
        )

    def execute(
        self,
        repo_id: str,
        path: str,
        revision: str = "main",
        max_chars: int = 200_000,
    ) -> str:
        try:
            repo_id = _valid_repo_id(repo_id)
            path = _valid_repo_path(path)
            revision = str(revision or "").strip()
            if not _REVISION_RE.fullmatch(revision) or ".." in revision.split("/"):
                raise ValueError("revision contains unsupported characters or traversal")
            if isinstance(max_chars, bool) or not 1 <= int(max_chars) <= 400_000:
                raise ValueError("max_chars must be an integer from 1 to 400000")
            max_chars = int(max_chars)
            canonical = (
                f"{_HF_ORIGIN}/{quote(repo_id, safe='/')}/resolve/"
                f"{quote(revision, safe='')}/{quote(path, safe='/')}"
            )
            body, response = _public_get(
                canonical,
                maximum=_MAX_FILE_BYTES,
                redirect_paths=(f"/{repo_id}/resolve/", "/api/resolve-cache/models/"),
            )
            try:
                text = body.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HuggingFacePublicAPIError("repository file is not UTF-8 text") from exc
            truncated = len(text) > max_chars
            result = {
                "source": canonical,
                "repo_commit": response.headers.get("x-repo-commit"),
                "etag": response.headers.get("etag") or response.headers.get("x-linked-etag"),
                "content": text[:max_chars],
                "truncated": truncated,
                "returned_characters": min(len(text), max_chars),
                "total_characters": len(text),
                "evidence_limits": [
                    "This is file content from the requested Hub revision; it does not independently validate factual claims inside the file.",
                    "License interpretation must include referenced upstream licenses, acceptable-use terms, and derivative redistribution conditions.",
                ],
            }
            return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        except (ValueError, TypeError, requests.RequestException, HuggingFacePublicAPIError) as exc:
            return _error(exc)
