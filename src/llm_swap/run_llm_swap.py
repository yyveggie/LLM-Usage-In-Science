#!/usr/bin/env python3
"""Compare estimated alpha across generation LLMs on the official corpus.

The real paper corpus (official monthly inference parquet) is fixed; only the
distribution Q changes per generation LLM. This script loads each model's
swapped distribution and estimates alpha on the same official inference months,
so the per-LLM trends can be compared side by side.

Include the special model name ``official`` to add the paper's original
gpt-3.5-based distribution as a baseline.

Output CSV columns: model, venue, month, sentence_count, estimated_alpha.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.reproduction_common import (
    RESULTS,
    distribution_file,
    estimate_alpha_for_file,
    load_official_distribution,
    month_file,
)
from llm_swap.fit_swapped_distribution import DEFAULT_SWAP_DIR


def resolve_distribution_path(model: str, venue: str, swap_dir: Path) -> Path:
    """Official baseline -> packaged file; otherwise the swapped per-LLM file."""
    if model == "official":
        return distribution_file(venue)
    return swap_dir / model / f"{venue}.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-LLM alpha comparison.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names (use 'official' for the paper's gpt-3.5 baseline).",
    )
    parser.add_argument("--venues", nargs="+", default=["CS"])
    parser.add_argument(
        "--months",
        nargs="+",
        default=["2022_11", "2023_6", "2024_2"],
        help="Official inference months to evaluate.",
    )
    parser.add_argument("--swap-dir", type=Path, default=DEFAULT_SWAP_DIR)
    parser.add_argument(
        "--output", type=Path, default=RESULTS / "llm_swap_comparison.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []

    for model in args.models:
        for venue in args.venues:
            dist_path = resolve_distribution_path(model, venue, args.swap_dir)
            if not dist_path.exists():
                print(f"[skip] missing distribution: {dist_path}")
                continue
            distribution = load_official_distribution(dist_path)

            for month in args.months:
                month_path = month_file(venue, month)
                if not month_path.exists():
                    print(f"[skip] missing inference month: {month_path}")
                    continue
                alpha, count = estimate_alpha_for_file(distribution, month_path)
                rows.append(
                    {
                        "model": model,
                        "venue": venue,
                        "month": month,
                        "sentence_count": count,
                        "estimated_alpha": alpha,
                    }
                )
                print(f"{model:14} {venue:7} {month}: alpha={alpha:.4f} (n={count})")

    if not rows:
        raise SystemExit("No results produced; check distributions and months.")

    # Reuse the shared CSV writer for consistent output formatting.
    from core.reproduction_common import write_csv

    write_csv(rows, args.output)
    print(f"\nSaved comparison to {args.output}")


if __name__ == "__main__":
    main()
