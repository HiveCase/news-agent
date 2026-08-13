"""
Command-line entry point.

Usage:
    python -m src.main --url https://example.com/some-news-article
    python -m src.main --file examples/sample_text.txt
    python -m src.main --text "Paste article text directly here..."

Prints a structured JSON response with the summary and sentiment.
"""
from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from .graph import run_agent

load_dotenv()  # load ANTHROPIC_API_KEY from a .env file if present


def main() -> None:
    parser = argparse.ArgumentParser(description="News summarization & sentiment agent")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="URL of the news article")
    group.add_argument("--text", help="Raw article text")
    group.add_argument("--file", help="Path to a .txt file with article text")
    args = parser.parse_args()

    raw_text = None
    if args.text:
        raw_text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            raw_text = f.read()

    result = run_agent(url=args.url, raw_text=raw_text)

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
