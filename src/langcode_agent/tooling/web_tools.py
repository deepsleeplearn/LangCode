from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

from langchain_tavily import TavilyExtract, TavilySearch


DEFAULT_SEARCH_DEPTH = "basic"
MAX_SEARCH_RESULTS = 10
MAX_FETCH_CHARS = 12000


def web_search(
    query: str,
    *,
    max_results: int = 5,
    search_depth: str = DEFAULT_SEARCH_DEPTH,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    topic: str = "general",
) -> dict:
    _require_tavily_key()
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("搜索关键词不能为空")

    tool = TavilySearch(
        max_results=max(1, min(int(max_results), MAX_SEARCH_RESULTS)),
        search_depth=_valid_search_depth(search_depth),
        include_domains=_clean_domains(include_domains),
        exclude_domains=_clean_domains(exclude_domains),
        topic=_valid_topic(topic),
        include_answer=False,
        include_raw_content=False,
    )
    result = tool.invoke({"query": clean_query})
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(str(result["error"]))
    return _normalize_search_result(result)


def web_fetch(
    url: str,
    *,
    extract_depth: str = DEFAULT_SEARCH_DEPTH,
    max_chars: int = MAX_FETCH_CHARS,
) -> dict:
    _require_tavily_key()
    safe_url = _validate_public_url(url)
    tool = TavilyExtract(
        extract_depth=_valid_extract_depth(extract_depth),
        format="markdown",
        include_images=False,
    )
    result = tool.invoke({"urls": [safe_url]})
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(str(result["error"]))
    return _normalize_fetch_result(result, max_chars=max_chars)


def _require_tavily_key() -> None:
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("未配置 TAVILY_API_KEY")


def _normalize_search_result(result: object) -> dict:
    if not isinstance(result, dict):
        return {"query": "", "results": []}

    normalized = []
    for item in result.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "content": str(item.get("content") or ""),
                "score": item.get("score"),
            }
        )
    return {
        "query": str(result.get("query") or ""),
        "results": normalized,
        "response_time": result.get("response_time"),
    }


def _normalize_fetch_result(result: object, *, max_chars: int) -> dict:
    if not isinstance(result, dict):
        return {"url": "", "content": ""}

    results = result.get("results") or []
    first = results[0] if results and isinstance(results[0], dict) else {}
    content = str(first.get("raw_content") or first.get("content") or "")
    limit = max(1000, min(int(max_chars), 50000))
    truncated = len(content) > limit
    if truncated:
        content = content[:limit]

    return {
        "url": str(first.get("url") or ""),
        "title": str(first.get("title") or ""),
        "content": content,
        "truncated": truncated,
        "failed_results": result.get("failed_results") or [],
    }


def _validate_public_url(url: str) -> str:
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("只能抓取 http 或 https URL")
    if not parsed.hostname:
        raise ValueError("URL 必须包含主机名")

    hostname = parsed.hostname.lower()
    if hostname in {"localhost"} or hostname.endswith(".local"):
        raise ValueError("不能抓取 localhost 或 .local URL")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return parsed.geturl()
    if address.is_private or address.is_loopback or address.is_link_local or address.is_multicast or address.is_reserved:
        raise ValueError("不能抓取私有、本地、组播或保留 IP URL")
    return parsed.geturl()


def _clean_domains(domains: list[str] | None) -> list[str] | None:
    if not domains:
        return None
    cleaned = [str(domain).strip() for domain in domains if str(domain).strip()]
    return cleaned or None


def _valid_search_depth(value: str) -> str:
    return value if value in {"basic", "advanced", "fast", "ultra-fast"} else DEFAULT_SEARCH_DEPTH


def _valid_extract_depth(value: str) -> str:
    return "advanced" if value == "advanced" else "basic"


def _valid_topic(value: str) -> str:
    return value if value in {"general", "news", "finance"} else "general"
