#!/usr/bin/env python3
"""Core-method entry for:

Quantifying large language model usage in scientific papers.

This Nature Human Behaviour entry uses the same official packaged data as the
authors' repository, with the extended September 2024 endpoint. By default it
estimates pre-ChatGPT reference month 2022_11 and endpoint month 2024_9 for
all venues. Use --all-months to compute the full monthly trend table.
"""

from __future__ import annotations

# Allow running directly from the project root without installing the package.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))

import argparse
from pathlib import Path

from core.reproduction_common import (
    RESULTS,
    VENUE_CONFIG,
    distribution_file,
    estimate_alpha_for_file,
    load_official_distribution,
    month_file,
    write_csv,
)


DEFAULT_VENUES = ["CS", "EESS", "Math", "Phys", "Stat", "bioRxiv", "Nature"]
DEFAULT_MONTHS = ["2022_11", "2024_9"]


def all_months_for_venue(venue: str) -> list[str]:
    inference_dir = VENUE_CONFIG[venue]["inference_dir"]
    return sorted(path.stem for path in inference_dir.glob("*.parquet"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper 3 core-method reproduction.")
    parser.add_argument("--venues", nargs="+", default=DEFAULT_VENUES)
    parser.add_argument("--months", nargs="+", default=DEFAULT_MONTHS)
    parser.add_argument(
        "--all-months",
        action="store_true",
        help="Estimate alpha for every downloaded month instead of --months.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "03_quantifying_usage_core.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str | int | float]] = []
    for venue in args.venues:
        model = load_official_distribution(distribution_file(venue))
        months = all_months_for_venue(venue) if args.all_months else args.months
        for month in months:
            path = month_file(venue, month)
            alpha, sentence_count = estimate_alpha_for_file(model, path)
            rows.append(
                {
                    "paper": "Quantifying large language model usage in scientific papers",
                    "venue": venue,
                    "month": month,
                    "file": str(path),
                    "sentence_count": sentence_count,
                    "estimated_alpha": alpha,
                }
            )
            print(f"{venue:7} {month}: alpha={alpha:.4f}, sentences={sentence_count}")

    write_csv(rows, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
