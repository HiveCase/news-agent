# News Summarization & Sentiment Agent

A small, easy-to-read AI agent built with **LangGraph**. It fetches a news
article (from a URL or pre-provided text), summarizes it with an LLM, runs
sentiment analysis, and returns a structured result.

## Pipeline

```
START ─► fetch ─► summarize ─► sentiment ─► END
```

| Step        | What it does                                   | Where           |
|-------------|------------------------------------------------|-----------------|
| `fetch`     | Get article text from URL or raw text (tool)   | `src/tools.py`  |
| `summarize` | 3–4 sentence summary via LLM (Claude)          | `src/nodes.py`  |
| `sentiment` | Rule-based lexicon sentiment (tool)            | `src/tools.py`  |

State is a single `AgentState` TypedDict (`src/state.py`) that flows
through every node.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then add your ANTHROPIC_API_KEY
```

> No API key? The agent still runs — it falls back to a simple
> extractive summary so you can test the whole pipeline offline. The
> sentiment tool is rule-based and never needs a key.

## Usage

```bash
# From a URL
python -m src.main --url "https://example.com/news-article"

# From a text file
python -m src.main --file examples/sample_text.txt

# From inline text
python -m src.main --text "Paste article text here..."
```

### Example output

```json
{
  "input": { "url": null, "used_raw_text": true },
  "summary": "NovaTech reported record quarterly earnings driven by cloud growth...",
  "sentiment": { "label": "POSITIVE", "score": 1.0 },
  "error": null
}
```

## Project layout

```
news_agent/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── state.py     # shared AgentState (TypedDict)
│   ├── tools.py     # fetch_news + analyze_sentiment tools
│   ├── nodes.py     # fetch / summarize / sentiment nodes
│   ├── graph.py     # LangGraph wiring + run_agent()
│   └── main.py      # CLI entry point
└── examples/
    ├── sample_text.txt        # positive sample
    └── sample_negative.txt    # negative sample
```

## Design notes

- **Why LangGraph?** The task is a clear multi-step pipeline with shared
  state. LangGraph makes each step an explicit node and the data flow
  easy to read, extend (e.g. add a branch), or debug.
- **Tools vs. LLM.** Fetching and sentiment are deterministic tools;
  only summarization uses the LLM. This keeps cost, latency, and
  reproducibility under control.
- **Graceful degradation.** Every node catches its own errors and writes
  to `state["error"]` instead of crashing the run.
