from langgraph.graph import StateGraph, START, END
from cpc_agent.models import AgentState
from cpc_agent.nodes.ingest import ingest_node
from cpc_agent.nodes.extract import extract_node
from cpc_agent.nodes.cluster import cluster_node
from cpc_agent.nodes.detect import detect_node
from cpc_agent.nodes.investigate import investigate_node
from cpc_agent.nodes.build_map import build_map_node


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("extract", extract_node)
    graph.add_node("cluster", cluster_node)
    graph.add_node("detect", detect_node)
    graph.add_node("investigate", investigate_node)
    graph.add_node("build_map", build_map_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "extract")
    graph.add_edge("extract", "cluster")
    graph.add_edge("cluster", "detect")
    graph.add_edge("detect", "investigate")
    graph.add_edge("investigate", "build_map")
    graph.add_edge("build_map", END)

    return graph.compile()
