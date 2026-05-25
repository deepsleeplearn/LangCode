from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from langchain_core.runnables.graph import Graph


MAX_MERMAID_CHARS = 12000
_ALLOWED_DIRECTIONS = {"TD", "TB", "BT", "LR", "RL"}
_ALLOWED_RAW_PREFIXES = (
    "graph ",
    "flowchart ",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "stateDiagram-v2",
    "erDiagram",
    "journey",
    "gantt",
    "mindmap",
    "timeline",
)
_DANGEROUS_MERMAID_PATTERNS = [
    (r"click\s+\w+\s+href\s+", "external_href"),
    (r"javascript:", "javascript_url"),
    (r"<\s*script", "script_tag"),
    (r"onerror\s*=", "html_event_handler"),
    (r"onload\s*=", "html_event_handler"),
]


def diagram_tool(workspace_root: str | Path, tool_input: dict) -> dict:
    """生成可在前端渲染的 Mermaid 图。

    结构化 flow/collaboration 图使用 LangChain Graph.draw_mermaid() 生成；
    sequence/state/class 等复杂图可传入 raw Mermaid DSL。
    """

    title = str(tool_input.get("title") or "流程图").strip()[:120]
    diagram_type = str(tool_input.get("diagram_type") or "flowchart").strip().lower()
    raw_mermaid = str(tool_input.get("mermaid") or "").strip()

    if raw_mermaid:
        mermaid = _normalize_raw_mermaid(raw_mermaid)
    else:
        mermaid = _build_graph_mermaid(tool_input)

    validation_error = _validate_mermaid(mermaid)
    if validation_error:
        return {"ok": False, "error": validation_error}

    return {
        "ok": True,
        "kind": "diagram",
        "title": title,
        "diagram_type": diagram_type,
        "format": "mermaid",
        "mermaid": mermaid,
        "content": mermaid,
        "message": "图已生成，可在前端渲染。",
    }


def _build_graph_mermaid(tool_input: dict) -> str:
    direction = str(tool_input.get("direction") or "TD").strip().upper()
    if direction not in _ALLOWED_DIRECTIONS:
        direction = "TD"
    nodes = _normalize_nodes(tool_input.get("nodes"))
    edges = _normalize_edges(tool_input.get("edges"))
    if not nodes:
        return f"flowchart {direction}\n  main[主 Agent]\n  user[用户]\n  user --> main"

    graph = Graph()
    graph_nodes = {}
    for node in nodes:
        graph_nodes[node["id"]] = graph.add_node(None, id=node["id"])

    for edge in edges:
        source = graph_nodes.get(edge["source"])
        target = graph_nodes.get(edge["target"])
        if source is None or target is None:
            continue
        graph.add_edge(source, target, edge.get("label") or None)

    mermaid = graph.draw_mermaid(with_styles=False)
    mermaid = _decode_langchain_node_ids(mermaid)
    mermaid = _replace_graph_header(mermaid, direction)
    mermaid = _apply_node_labels(mermaid, nodes)
    return mermaid


def _normalize_nodes(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    nodes: list[dict] = []
    seen: set[str] = set()
    for item in value[:40]:
        if not isinstance(item, dict):
            continue
        raw_id = str(item.get("id") or item.get("name") or "").strip()
        node_id = _safe_node_id(raw_id)
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        label = str(item.get("label") or item.get("name") or raw_id or node_id).strip()
        role = str(item.get("role") or "").strip()
        nodes.append({"id": node_id, "label": _safe_label(label), "role": _safe_label(role)})
    return nodes


def _normalize_edges(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    edges: list[dict] = []
    for item in value[:80]:
        if not isinstance(item, dict):
            continue
        source = _safe_node_id(str(item.get("source") or item.get("from") or ""))
        target = _safe_node_id(str(item.get("target") or item.get("to") or ""))
        if not source or not target:
            continue
        label = _safe_label(str(item.get("label") or item.get("action") or ""))
        edges.append({"source": source, "target": target, "label": label})
    return edges


def _normalize_raw_mermaid(mermaid: str) -> str:
    text = mermaid.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _validate_mermaid(mermaid: str) -> str | None:
    if not mermaid:
        return "Mermaid 内容不能为空"
    if len(mermaid) > MAX_MERMAID_CHARS:
        return f"Mermaid 内容超过 {MAX_MERMAID_CHARS} 字符"
    stripped = mermaid.lstrip()
    if not any(stripped.startswith(prefix) for prefix in _ALLOWED_RAW_PREFIXES):
        return "仅支持 Mermaid 图语法：graph、flowchart、sequenceDiagram、classDiagram、stateDiagram、erDiagram、journey、gantt、mindmap、timeline"
    for pattern, pattern_id in _DANGEROUS_MERMAID_PATTERNS:
        if re.search(pattern, mermaid, re.IGNORECASE):
            return f"Mermaid 内容包含不安全片段：{pattern_id}"
    return None


def _safe_node_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        return ""
    if normalized[0].isdigit():
        normalized = f"n_{normalized}"
    return normalized[:64]


def _safe_label(value: str) -> str:
    text = " ".join(str(value or "").split())
    text = text.replace('"', "'").replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    return text[:120]


def _decode_langchain_node_ids(mermaid: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        try:
            return bytes(raw.replace("\\", ""), "ascii").decode("unicode_escape")
        except UnicodeDecodeError:
            return raw

    return re.sub(r"(?:\\[0-9a-fA-F]{2})+", replace, mermaid)


def _replace_graph_header(mermaid: str, direction: str) -> str:
    return re.sub(r"^graph\s+\w+;", f"flowchart {direction}", mermaid.strip(), count=1)


def _apply_node_labels(mermaid: str, nodes: list[dict]) -> str:
    lines = [line.rstrip(";") for line in mermaid.splitlines()]
    present = set()
    for line in lines:
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\b", line)
        if match:
            present.add(match.group(1))
    additions = []
    for node in nodes:
        if node["id"] in present:
            additions.append(f'  {node["id"]}["{node["label"] or node["id"]}"]')
    if additions:
        return "\n".join(lines + additions)
    return "\n".join(lines)
