#!/usr/bin/env python3
"""Build a per-LLM distribution by swapping only Q (q_t).

The mixture model needs P (human) and Q (AI) over the *same* token set T.
Swapping the generation LLM should change only Q. This script therefore
supports two modes:

- ``shared-human`` (default, recommended): take the vocabulary T and the human
  side (logP, log1-P) directly from an official venue distribution parquet, and
  recompute only the AI side (logQ, log1-Q) from the new model's AI corpus.
  This isolates the LLM effect and keeps results comparable to the paper and
  across models.

- ``from-scratch``: fit both P and Q from supplied human and AI sentence
  corpora (uses the repo's reference estimator). Use this if you also want to
  rebuild the human side.

Output matches the official schema (Word, logP, logQ, log1-P, log1-Q), so it
plugs straight into ``reproduction_common.load_official_distribution``,
``run_02/run_03``, and ``official_cs_validation.py`` (via --distribution).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from core.distributional_llm_reproduction import (
    count_binary_occurrences,
    fit_distribution_model,
)
from core.paths import RESULTS
from core.reproduction_common import distribution_file


DEFAULT_SWAP_DIR = RESULTS / "swapped_distribution"
REQUIRED_COLUMNS = ["Word", "logP", "logQ", "log1-P", "log1-Q"]


def load_ai_sentences(path: Path) -> list[list[str]]:
    df = pd.read_parquet(path)
    if "inference_sentence" not in df.columns:
        raise ValueError(f"{path} has no inference_sentence column")
    return [list(sentence) for sentence in df["inference_sentence"] if len(sentence) > 1]


def recompute_q_over_vocabulary(
    official_distribution: Path,
    ai_sentences: list[list[str]],
    *,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    """Keep official human side; recompute logQ/log1-Q from new AI sentences."""
    df = pd.read_parquet(official_distribution)
    missing = {"Word", "logP", "log1-P"}.difference(df.columns)
    if missing:
        raise ValueError(f"{official_distribution} missing columns: {sorted(missing)}")

    ai_counts = count_binary_occurrences(ai_sentences)
    n_ai = len(ai_sentences)
    denom = n_ai + 2.0 * smoothing

    log_q: list[float] = []
    log_not_q: list[float] = []
    for word in df["Word"].astype(str):
        q = (ai_counts.get(word, 0) + smoothing) / denom
        log_q.append(math.log(q))
        log_not_q.append(math.log1p(-q))

    out = df[["Word", "logP", "log1-P"]].copy()
    out["logQ"] = log_q
    out["log1-Q"] = log_not_q
    return out[REQUIRED_COLUMNS]


def fit_from_scratch(
    human_sentences: list[list[str]],
    ai_sentences: list[list[str]],
    *,
    min_occurrences: int,
) -> pd.DataFrame:
    model = fit_distribution_model(
        human_sentences,
        ai_sentences,
        min_human_occurrences=min_occurrences,
        min_ai_occurrences=min_occurrences,
    )
    rows = [
        {
            "Word": word,
            "logP": model.log_p[word],
            "logQ": model.log_q[word],
            "log1-P": model.log_not_p[word],
            "log1-Q": model.log_not_q[word],
        }
        for word in model.vocabulary
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a per-LLM swapped distribution.")
    parser.add_argument("--model", required=True, help="Model name (for output path).")
    parser.add_argument("--venue", default="CS", help="Official venue for human side.")
    parser.add_argument(
        "--ai-corpus",
        type=Path,
        required=True,
        help="AI sentences parquet from build_ai_corpus.py.",
    )
    parser.add_argument(
        "--mode",
        choices=["shared-human", "from-scratch"],
        default="shared-human",
    )
    parser.add_argument(
        "--official-distribution",
        type=Path,
        default=None,
        help="Override official distribution path (default: venue's official file).",
    )
    parser.add_argument(
        "--human-sentences",
        type=Path,
        default=None,
        help="Human sentences parquet (required for from-scratch mode).",
    )
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--min-occurrences", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet. Default: results/swapped_distribution/<model>/<venue>.parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ai_sentences = load_ai_sentences(args.ai_corpus)
    print(f"Loaded {len(ai_sentences)} AI sentences from {args.ai_corpus}")

    if args.mode == "shared-human":
        official = args.official_distribution or distribution_file(args.venue)
        result = recompute_q_over_vocabulary(
            official, ai_sentences, smoothing=args.smoothing
        )
        print(f"Reused human side from {official}; recomputed Q for {len(result)} words.")
    else:
        if not args.human_sentences:
            raise SystemExit("--human-sentences is required for from-scratch mode.")
        human_sentences = load_ai_sentences(args.human_sentences)
        result = fit_from_scratch(
            human_sentences, ai_sentences, min_occurrences=args.min_occurrences
        )
        print(f"Fitted P and Q from scratch over {len(result)} words.")

    output = args.output or (DEFAULT_SWAP_DIR / args.model / f"{args.venue}.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    print(f"Saved swapped distribution to {output}")


if __name__ == "__main__":
    main()
