#!/usr/bin/env python3
"""Collect genuine pre-ChatGPT human paragraphs from arXiv.

The "swap the generation LLM" experiment needs *real* human-written paragraphs
(with punctuation, mixed case) to seed the two-stage generation. The official
packaged data only ships tokenized/lowercased sentences, so we collect raw
abstracts directly from the arXiv API.

Abstracts published before the ChatGPT launch (2020-01 .. 2022-11-29) are
human-written by construction and are the same kind of text the original papers
sampled. Output is JSONL compatible with ``human_corpus.py --source jsonl``:

    {"text": <abstract>, "arxiv_id": ..., "title": ..., "published": ..., "category": ...}

Standard library only (urllib + xml.etree), so no extra dependencies.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from core.paths import DATA_DIR


ARXIV_API = "http://export.arxiv.org/api/query"

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# CS subcategories analysed in the papers; override with --categories.
DEFAULT_CATEGORIES = ["cs.CL", "cs.LG", "cs.CV"]
DEFAULT_OUTPUT = DATA_DIR / "human_corpus" / "arxiv_human.jsonl"


def normalize_text(text: str) -> str:
    """Collapse arXiv abstract whitespace/newlines into one clean paragraph."""
    return " ".join(text.split()).strip()


def build_query(categories: list[str], start_date: str, end_date: str) -> str:
    cat_clause = " OR ".join(f"cat:{cat}" for cat in categories)
    date_clause = f"submittedDate:[{start_date} TO {end_date}]"
    return f"({cat_clause}) AND {date_clause}"


def fetch_page(query: str, start: int, page_size: int, timeout: float) -> bytes:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
    )
    url = f"{ARXIV_API}?{params}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def parse_entries(xml_bytes: bytes, min_chars: int) -> list[dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    rows: list[dict[str, str]] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        summary_el = entry.find(f"{ATOM_NS}summary")
        if summary_el is None or not summary_el.text:
            continue
        abstract = normalize_text(summary_el.text)
        if len(abstract) < min_chars:
            continue

        id_el = entry.find(f"{ATOM_NS}id")
        title_el = entry.find(f"{ATOM_NS}title")
        published_el = entry.find(f"{ATOM_NS}published")
        primary_el = entry.find(f"{ARXIV_NS}primary_category")

        rows.append(
            {
                "text": abstract,
                "arxiv_id": (id_el.text or "").strip() if id_el is not None else "",
                "title": normalize_text(title_el.text) if title_el is not None and title_el.text else "",
                "published": (published_el.text or "").strip() if published_el is not None else "",
                "category": primary_el.get("term", "") if primary_el is not None else "",
            }
        )
    return rows


def collect(args: argparse.Namespace) -> int:
    query = build_query(args.categories, args.start_date, args.end_date)
    print(f"Query: {query}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    start = 0

    with args.output.open("w", encoding="utf-8") as handle:
        while written < args.max_papers:
            page_size = min(args.page_size, args.max_papers - written)
            try:
                xml_bytes = fetch_page(query, start, page_size, args.timeout)
            except Exception as exc:  # noqa: BLE001 - network resilience
                print(f"[warn] fetch failed at start={start}: {exc}; retrying once")
                time.sleep(args.sleep * 2)
                try:
                    xml_bytes = fetch_page(query, start, page_size, args.timeout)
                except Exception as exc2:  # noqa: BLE001
                    print(f"[error] giving up at start={start}: {exc2}")
                    break

            entries = parse_entries(xml_bytes, args.min_chars)
            if not entries:
                print("No more entries returned; stopping.")
                break

            for row in entries:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += len(entries)
            start += page_size
            print(f"  collected {written}/{args.max_papers}")

            # arXiv API etiquette: wait between requests.
            time.sleep(args.sleep)

    print(f"Saved {written} human paragraphs to {args.output}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect pre-ChatGPT human abstracts from arXiv as generation seeds."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="arXiv categories, e.g. cs.CL cs.LG (no wildcards).",
    )
    parser.add_argument(
        "--start-date",
        default="200001010000",
        help="Submitted-date lower bound, YYYYMMDDHHMM.",
    )
    parser.add_argument(
        "--end-date",
        default="202211290000",
        help="Submitted-date upper bound, YYYYMMDDHHMM (default = ChatGPT launch).",
    )
    parser.add_argument("--max-papers", type=int, default=500)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    collect(parse_args())


if __name__ == "__main__":
    main()
