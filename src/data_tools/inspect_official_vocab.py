#!/usr/bin/env python3
"""Check how well a tokenizer aligns with the official vocabulary.

If the tokenizer used to build the AI corpus does not match the official
tokenization, q_t is biased and so is alpha. This tool quantifies the overlap:
it tokenizes real human paragraphs with each available tokenizer and reports how
much of the output lands in the official ``Word`` vocabulary for a venue.

Higher coverage = better alignment. Compare ``regex`` vs ``spacy`` and pick the
one with higher coverage (spaCy is expected to match the official pipeline).
"""

from __future__ import annotations

import argparse

import pandas as pd

from core.reproduction_common import distribution_file
from core.tokenization import TOKENIZER_CHOICES, get_tokenizer
from llm_swap.human_corpus import add_source_arguments, load_human_paragraphs


def load_official_vocab(venue: str) -> set[str]:
    df = pd.read_parquet(distribution_file(venue))
    return set(df["Word"].astype(str))


def coverage(paragraphs: list[str], tokenize, vocab: set[str]) -> dict[str, float]:
    types: set[str] = set()
    total = 0
    in_vocab_occ = 0
    for paragraph in paragraphs:
        for token in tokenize(paragraph):
            types.add(token)
            total += 1
            if token in vocab:
                in_vocab_occ += 1
    in_vocab_types = sum(1 for t in types if t in vocab)
    return {
        "token_types": len(types),
        "token_occurrences": total,
        "type_coverage": in_vocab_types / max(1, len(types)),
        "occurrence_coverage": in_vocab_occ / max(1, total),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure tokenizer overlap with the official venue vocabulary."
    )
    add_source_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab = load_official_vocab(args.venue)
    print(f"Official '{args.venue}' vocabulary size: {len(vocab)}")
    print("Sample official words:", sorted(list(vocab))[:15])

    paragraphs = load_human_paragraphs(args)
    print(f"\nTokenizing {len(paragraphs)} human paragraphs with each tokenizer:\n")

    for name in TOKENIZER_CHOICES:
        try:
            tokenize = get_tokenizer(name)
            stats = coverage(paragraphs, tokenize, vocab)
        except Exception as exc:  # noqa: BLE001 - e.g. spaCy not installed
            print(f"  {name:6}: unavailable ({exc})")
            continue
        print(
            f"  {name:6}: type_coverage={stats['type_coverage']:.3f} "
            f"occurrence_coverage={stats['occurrence_coverage']:.3f} "
            f"(types={stats['token_types']}, tokens={stats['token_occurrences']})"
        )

    print(
        "\nHigher coverage = better alignment. Use the winning tokenizer in "
        "build_ai_corpus (--tokenizer)."
    )


if __name__ == "__main__":
    main()
