"""
The three worker nodes of the graph.

Each node is a pure function: state in -> partial state out. LangGraph
merges the returned dict back into the shared state.

    fetch_node      -> puts article_text into state
    summarize_node  -> puts summary into state (uses the LLM)
    sentiment_node  -> puts sentiment + sentiment_score into state
"""
from __future__ import annotations

import os

from .state import AgentState
from .tools import fetch_news, analyze_sentiment


# --------------------------------------------------------------------------
# Node 1: fetch
# --------------------------------------------------------------------------
def fetch_node(state: AgentState) -> dict:
    try:
        text = fetch_news(url=state.get("url"), raw_text=state.get("raw_text"))
        return {"article_text": text}
    except Exception as e:  # noqa: BLE001  (surface any fetch failure cleanly)
        return {"error": f"fetch failed: {e}"}


# --------------------------------------------------------------------------
# Node 2: summarize (LLM)
# --------------------------------------------------------------------------
SUMMARY_PROMPT = (
    "You are a concise news editor. Summarize the article below in 3-4 "
    "sentences. Capture the key facts only; do not add opinions.\n\n"
    "ARTICLE:\n{article}\n\nSUMMARY:"
)


def _get_llm():
    """Create the chat model. Returns None if no API key is configured,
    which lets the agent fall back to a simple extractive summary so the
    pipeline is always runnable."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest"),
        temperature=0,
        max_tokens=400,
    )


def _extractive_fallback(text: str, n: int = 3) -> str:
    """Very small fallback: return the first n sentences. Only used when
    no LLM API key is present, so the demo still runs offline."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:n]).strip()


def summarize_node(state: AgentState) -> dict:
    if state.get("error"):
        return {}  # short-circuit: an earlier step already failed

    article = state["article_text"]
    llm = _get_llm()

    if llm is None:
        return {"summary": _extractive_fallback(article)}

    try:
        msg = llm.invoke(SUMMARY_PROMPT.format(article=article[:6000]))
        return {"summary": msg.content.strip()}
    except Exception as e:  # noqa: BLE001
        # Fall back rather than crash the whole run.
        return {"summary": _extractive_fallback(article),
                "error": f"llm summary failed, used fallback: {e}"}


# --------------------------------------------------------------------------
# Node 3: sentiment
# --------------------------------------------------------------------------
def sentiment_node(state: AgentState) -> dict:
    if state.get("error") and not state.get("summary"):
        return {}

    # Analyze the full article for the most representative signal.
    result = analyze_sentiment(state.get("article_text", ""))
    return {"sentiment": result["label"], "sentiment_score": result["score"]}
