"""
Assemble the LangGraph.

The flow is a simple linear pipeline:

    START -> fetch -> summarize -> sentiment -> END

LangGraph handles passing the shared state between nodes. We expose a
`build_graph()` factory and a `run_agent()` convenience wrapper that
returns a clean, structured result.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import fetch_node, summarize_node, sentiment_node


def build_graph():
    """Construct and compile the agent graph."""
    workflow = StateGraph(AgentState)

    # Register the three nodes.
    workflow.add_node("fetch", fetch_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("sentiment", sentiment_node)

    # Wire them in order.
    workflow.add_edge(START, "fetch")
    workflow.add_edge("fetch", "summarize")
    workflow.add_edge("summarize", "sentiment")
    workflow.add_edge("sentiment", END)

    return workflow.compile()


def run_agent(url: str | None = None, raw_text: str | None = None) -> dict:
    """Run the full pipeline and return a structured response.

    Exactly one of `url` or `raw_text` should be supplied.
    """
    graph = build_graph()
    initial: AgentState = {"url": url, "raw_text": raw_text}
    final = graph.invoke(initial)

    # Shape the output into a clean, predictable structure.
    return {
        "input": {"url": url, "used_raw_text": bool(raw_text)},
        "summary": final.get("summary", ""),
        "sentiment": {
            "label": final.get("sentiment", "NEUTRAL"),
            "score": final.get("sentiment_score", 0.0),
        },
        "error": final.get("error"),
    }
