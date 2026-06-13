#!/usr/bin/env python3
"""Collect pre-ChatGPT human paragraphs from bioRxiv.

Biomedical counterpart of ``collect_arxiv_human.py`` for the third paper's
domain (PubMed Central / bioRxiv). It pulls real human-written abstracts from
the official bioRxiv API for a date range before the ChatGPT launch, to seed
the two-stage AI generation.

Output is JSONL compatible with ``human_corpus.py --source jsonl``:

    {"text": <abstract>, "doi": ..., "title": ..., "date": ..., "category": ...}

Standard library only (urllib + json).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from core.paths import DATA_DIR


# Official bioRxiv API; returns up to 100 records per cursor step.
API_TEMPLATE = "https://api.biorxiv.org/details/biorxiv/{start}/{end}/{cursor}"
PAGE_SIZE = 100
DEFAULT_OUTPUT = DATA_DIR / "human_corpus" / "biorxiv_human.jsonl"


def normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def fetch_page(start: str, end: str, cursor: int, timeout: float) -> dict:
    url = API_TEMPLATE.format(start=start, end=end, cursor=cursor)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def collect(args: argparse.Namespace) -> int:
    categories = {c.lower() for c in args.categories} if args.categories else None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    cursor = 0
    total = None

    with args.output.open("w", encoding="utf-8") as handle:
        while written < args.max_papers:
            try:
                payload = fetch_page(args.start_date, args.end_date, cursor, args.timeout)
            except Exception as exc:  # noqa: BLE001 - network resilience
                print(f"[warn] fetch failed at cursor={cursor}: {exc}; retrying once")
                time.sleep(args.sleep * 2)
                try:
                    payload = fetch_page(args.start_date, args.end_date, cursor, args.timeout)
                except Exception as exc2:  # noqa: BLE001
                    print(f"[error] giving up at cursor={cursor}: {exc2}")
                    break

            messages = payload.get("messages", [{}])
            if total is None:
                total = messages[0].get("total")
                print(f"bioRxiv reports total={total} papers in range.")

            collection = payload.get("collection", [])
            if not collection:
                print("No more records returned; stopping.")
                break

            for item in collection:
                abstract = normalize_text(item.get("abstract", ""))
                if len(abstract) < args.min_chars:
                    continue
                category = (item.get("category") or "").strip()
                if categories is not None and category.lower() not in categories:
                    continue
                handle.write(
                    json.dumps(
                        {
                            "text": abstract,
                            "doi": (item.get("doi") or "").strip(),
                            "title": normalize_text(item.get("title", "")),
                            "date": (item.get("date") or "").strip(),
                            "category": category,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
                if written >= args.max_papers:
                    break

            cursor += PAGE_SIZE
            print(f"  collected {written}/{args.max_papers} (cursor={cursor})")
            time.sleep(args.sleep)

    print(f"Saved {written} human paragraphs to {args.output}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect pre-ChatGPT human abstracts from bioRxiv as generation seeds."
    )
    parser.add_argument("--start-date", default="2020-01-01", help="YYYY-MM-DD.")
    parser.add_argument(
        "--end-date",
        default="2022-11-29",
        help="YYYY-MM-DD (default = day before ChatGPT launch).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional client-side filter on bioRxiv category (e.g. neuroscience).",
    )
    parser.add_argument("--max-papers", type=int, default=500)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    collect(parse_args())


if __name__ == "__main__":
    main()
