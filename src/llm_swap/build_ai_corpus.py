#!/usr/bin/env python3
"""Generate an AI-modified corpus with a chosen LLM (the swappable step).

For each human seed paragraph, this runs the paper's two-stage generation
(summarize -> expand) with the selected model, splits the output into
sentences, tokenizes them with the same tokenizer used elsewhere in this
repo, and saves one AI sentence per row.

Output schema (parquet):
    inference_sentence : list[str]   # tokenized AI sentence
    source_paragraph   : int         # index of the seeding human paragraph

The output parquet is directly consumable by ``fit_swapped_distribution.py``.

Note: this script performs live API calls and therefore requires the relevant
API key to be exported and the ``openai`` package installed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from core.distributional_llm_reproduction import tokenize
from core.paths import RESULTS
from llm_swap.human_corpus import add_source_arguments, load_human_paragraphs
from llm_swap.llm_models import DEFAULT_CONFIG, LLMClient, get_model_config


DEFAULT_AI_CORPUS_DIR = RESULTS / "ai_corpus"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (stdlib only)."""
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def generate_ai_sentences(
    client: LLMClient,
    paragraphs: list[str],
    *,
    strategy: str,
    min_tokens: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, paragraph in enumerate(paragraphs):
        try:
            ai_text = client.generate(paragraph, strategy=strategy)
        except Exception as exc:  # noqa: BLE001 - skip a failed paragraph
            print(f"[warn] paragraph {index} failed: {exc}")
            continue

        for sentence in split_sentences(ai_text):
            tokens = tokenize(sentence)
            if len(tokens) >= min_tokens:
                rows.append({"inference_sentence": tokens, "source_paragraph": index})

        if (index + 1) % 20 == 0:
            print(f"  generated {index + 1}/{len(paragraphs)} paragraphs")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an AI-modified corpus with a configured LLM."
    )
    parser.add_argument("--model", required=True, help="Model name from models.yaml.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--strategy",
        choices=["two_stage", "proofread"],
        default="two_stage",
        help="Generation strategy (paper main method vs proofread robustness).",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=2,
        help="Drop generated sentences shorter than this many tokens.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path. Defaults to results/ai_corpus/<model>_<venue>.parquet.",
    )
    add_source_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paragraphs = load_human_paragraphs(args)
    print(f"Loaded {len(paragraphs)} human seed paragraphs.")

    config = get_model_config(args.model, args.config)
    client = LLMClient(config)
    print(f"Generating with model {config.name!r} ({config.model}).")

    rows = generate_ai_sentences(
        client,
        paragraphs,
        strategy=args.strategy,
        min_tokens=args.min_tokens,
    )
    if not rows:
        raise SystemExit("No AI sentences generated; nothing to save.")

    output = args.output or (
        DEFAULT_AI_CORPUS_DIR / f"{args.model}_{args.venue}.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    print(f"Saved {len(rows)} AI sentences to {output}")


if __name__ == "__main__":
    main()
