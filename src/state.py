"""
Shared state for the news-summarization agent.

Every node in the LangGraph reads from and writes to this single
dictionary-like object. Using a TypedDict keeps the data flow explicit
and easy to follow.
"""
from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    # ---- input (one of these is provided by the user) ----
    url: Optional[str]            # a news article URL
    raw_text: Optional[str]       # or pre-provided article text

    # ---- produced along the pipeline ----
    article_text: str             # text after the fetch step
    summary: str                  # LLM-generated summary
    sentiment: str                # POSITIVE / NEGATIVE / NEUTRAL
    sentiment_score: float        # signed score in [-1, 1]
    error: Optional[str]          # populated if a step fails
