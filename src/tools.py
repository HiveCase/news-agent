"""
Tools used by the agent.

Two tools live here:

1. fetch_news_tool  -> gets article text from a URL (or passes text through)
2. sentiment_tool   -> rule-based (lexicon) sentiment analysis

Both are plain, well-documented functions. They are wrapped as LangChain
tools at the bottom so they can also be called by an LLM if desired, but
the graph calls the underlying functions directly for simplicity.
"""
from __future__ import annotations

import re
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


# --------------------------------------------------------------------------
# Tool 1: Fetch news
# --------------------------------------------------------------------------
def fetch_news(url: str | None = None, raw_text: str | None = None) -> str:
    """Return clean article text.

    If `raw_text` is given, it is returned as-is (cleaned of extra
    whitespace). Otherwise the `url` is downloaded and the visible
    paragraph text is extracted with BeautifulSoup.
    """
    if raw_text and raw_text.strip():
        return _clean(raw_text)

    if not url:
        raise ValueError("Provide either a URL or raw_text to fetch_news.")

    headers = {"User-Agent": "Mozilla/5.0 (news-agent/1.0)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Drop non-content tags.
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Prefer <article> if present, else join all paragraphs.
    article = soup.find("article")
    container = article if article else soup
    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = " ".join(paragraphs) if paragraphs else soup.get_text(" ", strip=True)

    text = _clean(text)
    if len(text) < 40:
        raise ValueError("Could not extract meaningful article text from the URL.")
    return text


def _clean(text: str) -> str:
    """Collapse whitespace to keep things tidy for the LLM."""
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Tool 2: Sentiment analysis (rule-based lexicon)
# --------------------------------------------------------------------------
# A small, transparent lexicon. Easy to read and to extend.
POSITIVE_WORDS = {
    "good", "great", "excellent", "positive", "success", "successful", "win",
    "wins", "won", "gain", "gains", "growth", "grow", "surge", "soar", "boost",
    "record", "strong", "improve", "improved", "improvement", "profit", "hope",
    "hopeful", "breakthrough", "celebrate", "praise", "optimistic", "rise",
    "rising", "recover", "recovery", "benefit", "advance", "opportunity",
}

NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "negative", "fail", "failed", "failure", "loss",
    "losses", "lose", "lost", "decline", "drop", "fall", "falls", "crash",
    "crisis", "weak", "worse", "worst", "risk", "threat", "concern", "concerns",
    "fear", "fears", "warn", "warning", "cut", "cuts", "layoff", "layoffs",
    "recession", "damage", "danger", "conflict", "protest", "violence", "death",
    "dead", "attack", "collapse", "shortage", "scandal", "controversy",
}

NEGATIONS = {"not", "no", "never", "n't", "without", "hardly", "barely"}


def analyze_sentiment(text: str) -> dict:
    """Return {'label': ..., 'score': ...}.

    The method: tokenize, count positive/negative lexicon hits, flip a hit
    when the previous token is a negation, then normalize into a score in
    [-1, 1] and map that to a label.
    """
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    pos = neg = 0

    for i, tok in enumerate(tokens):
        negated = i > 0 and tokens[i - 1] in NEGATIONS
        if tok in POSITIVE_WORDS:
            neg += 1 if negated else 0
            pos += 0 if negated else 1
        elif tok in NEGATIVE_WORDS:
            pos += 1 if negated else 0
            neg += 0 if negated else 1

    total = pos + neg
    score = 0.0 if total == 0 else round((pos - neg) / total, 3)

    if score > 0.15:
        label = "POSITIVE"
    elif score < -0.15:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {"label": label, "score": score}


# --------------------------------------------------------------------------
# LangChain tool wrappers (optional, for LLM tool-calling / demonstration)
# --------------------------------------------------------------------------
@tool
def fetch_news_tool(url: str) -> str:
    """Fetch and return the plain text of a news article from a URL."""
    return fetch_news(url=url)


@tool
def sentiment_tool(text: str) -> dict:
    """Analyze sentiment of text; returns a label and a score in [-1, 1]."""
    return analyze_sentiment(text)
