"""LangGraph StateGraph for the Adversarial Reviewer Agent."""

from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import ingest_node, classify_node, plan_node, check_node, report_node


def _route_after_ingest(state: AgentState) -> str:
    return "report" if state.get("status") == "error" else "classify"


def _route_after_check(state: AgentState) -> str:
    if state.get("status") in ("done", "error"):
        return "report"
    completed = state.get("checks_completed", [])
    planned = state.get("planned_checks", [])
    return "check" if any(c not in completed for c in planned) else "report"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("ingest", ingest_node)
    builder.add_node("classify", classify_node)
    builder.add_node("plan", plan_node)
    builder.add_node("check", check_node)
    builder.add_node("report", report_node)

    builder.add_edge(START, "ingest")
    builder.add_conditional_edges(
        "ingest",
        _route_after_ingest,
        {"classify": "classify", "report": "report"},
    )
    builder.add_edge("classify", "plan")
    builder.add_edge("plan", "check")
    builder.add_conditional_edges(
        "check",
        _route_after_check,
        {"check": "check", "report": "report"},
    )
    builder.add_edge("report", END)

    return builder.compile()
