#!/usr/bin/env python3
"""Load human reference paragraphs that seed the two-stage AI generation.

To rebuild the AI distribution Q for a new LLM, we need pre-ChatGPT *human*
paragraphs to feed into the counterfactual "summarize -> expand" generation.
The official packaged data does not ship raw human paragraphs, so this module
supports three sources:

1. ``jsonl``     - one JSON object per line with a text field (recommended;
                   bring your own pre-2023 paragraphs).
2. ``txt``       - paragraphs separated by blank lines.
3. ``official``  - reconstruct (detokenize) paragraphs from the official
                   pre-ChatGPT monthly inference parquet files. These months
                   are ~99% human, so they are a reasonable fallback seed.

All loaders return a list[str] of paragraphs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from core.reproduction_common import month_file


# Months safely before the ChatGPT launch (2022-11-30); ~pure human text.
DEFAULT_HUMAN_MONTHS = ["2021_1", "2021_6", "2022_1", "2022_6"]


def load_paragraphs_jsonl(path: Path, text_field: str = "text") -> list[str]:
    paragraphs: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get(text_field, "")).strip()
            if text:
                paragraphs.append(text)
    return paragraphs


def load_paragraphs_txt(path: Path) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    blocks = [block.strip() for block in raw.split("\n\n")]
    return [block for block in blocks if block]


def _detokenize(tokens: list[str]) -> str:
    return " ".join(str(token) for token in tokens)


def sample_paragraphs_from_official(
    venue: str,
    months: list[str],
    *,
    limit: int,
    seed: int = 13,
) -> list[str]:
    """Rebuild approximate human paragraphs from official inference parquet.

    Each official inference row is one abstract = a list of tokenized
    sentences. We detokenize each sentence and join them into a paragraph.
    The result is lowercase and punctuation-free, but still usable as a
    generation seed.
    """
    paragraphs: list[str] = []
    for month in months:
        path = month_file(venue, month)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "inference_sentence" not in df.columns:
            continue
        for value in df["inference_sentence"]:
            if value is None:
                continue
            items = list(value)
            if not items:
                continue
            first = items[0]
            if isinstance(first, str):
                # Row is a single tokenized sentence.
                paragraphs.append(_detokenize(items))
            else:
                # Row is a list of tokenized sentences (one abstract).
                sentences = [_detokenize(list(sentence)) for sentence in items]
                paragraphs.append(". ".join(sentences))

    rng = random.Random(seed)
    rng.shuffle(paragraphs)
    return paragraphs[:limit] if limit > 0 else paragraphs


def load_human_paragraphs(args: argparse.Namespace) -> list[str]:
    """Dispatch on ``args.source`` and return a list of paragraphs."""
    if args.source == "jsonl":
        paragraphs = load_paragraphs_jsonl(args.input, args.text_field)
    elif args.source == "txt":
        paragraphs = load_paragraphs_txt(args.input)
    elif args.source == "official":
        paragraphs = sample_paragraphs_from_official(
            args.venue, args.months, limit=args.limit, seed=args.seed
        )
    else:
        raise ValueError(f"Unknown source: {args.source!r}")

    if args.limit > 0:
        paragraphs = paragraphs[: args.limit]
    if not paragraphs:
        raise ValueError("No human paragraphs loaded; check the source arguments.")
    return paragraphs


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared CLI flags so other scripts can reuse the same source selection."""
    parser.add_argument(
        "--source",
        choices=["jsonl", "txt", "official"],
        default="official",
        help="Where to read human seed paragraphs from.",
    )
    parser.add_argument("--input", type=Path, help="Path for jsonl/txt sources.")
    parser.add_argument("--text-field", default="text", help="JSONL text field.")
    parser.add_argument(
        "--venue", default="CS", help="Venue for the official source."
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=DEFAULT_HUMAN_MONTHS,
        help="Pre-ChatGPT months for the official source.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max paragraphs to load (0 = no limit).",
    )
    parser.add_argument("--seed", type=int, default=13)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect human seed paragraphs.")
    add_source_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    paras = load_human_paragraphs(parse_args())
    print(f"Loaded {len(paras)} human paragraphs.")
    for preview in paras[:3]:
        print("-", preview[:160])
