"""LangGraph StateGraph definition for the Citation Genealogy Agent."""

from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import ingest_node, fetch_node, analyze_node, report_node


def _route(state: AgentState) -> str:
    if state.get("status") == "done":
        return "report"
    queue = state.get("queue") or []
    if not queue:
        return "report"
    max_depth = state.get("max_depth", 5)
    if all(item.get("depth", 0) >= max_depth for item in queue):
        return "report"
    return "fetch"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("ingest", ingest_node)
    builder.add_node("fetch", fetch_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("report", report_node)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "fetch")
    builder.add_edge("fetch", "analyze")
    builder.add_conditional_edges(
        "analyze",
        _route,
        {"fetch": "fetch", "report": "report"},
    )
    builder.add_edge("report", END)

    return builder.compile()
