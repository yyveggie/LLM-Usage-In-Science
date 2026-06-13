#!/usr/bin/env python3
"""Core-method entry for:

Mapping the Increasing Use of LLMs in Scientific Papers.

This conference-paper entry estimates alpha for the paper's core monthly
comparison around the February 2024 endpoint, using the official packaged
distribution and inference parquet files.
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
    distribution_file,
    estimate_alpha_for_file,
    load_official_distribution,
    month_file,
    write_csv,
)


DEFAULT_VENUES = ["CS", "EESS", "Math", "Phys", "Stat", "bioRxiv", "Nature"]
DEFAULT_MONTHS = ["2022_11", "2024_2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper 2 core-method reproduction.")
    parser.add_argument("--venues", nargs="+", default=DEFAULT_VENUES)
    parser.add_argument("--months", nargs="+", default=DEFAULT_MONTHS)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "02_mapping_increasing_use_core.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str | int | float]] = []
    for venue in args.venues:
        model = load_official_distribution(distribution_file(venue))
        for month in args.months:
            path = month_file(venue, month)
            alpha, sentence_count = estimate_alpha_for_file(model, path)
            rows.append(
                {
                    "paper": "Mapping the Increasing Use of LLMs in Scientific Papers",
                    "venue": venue,
                    "month": month,
                    "file": str(path.relative_to(path.parents[3])),
                    "sentence_count": sentence_count,
                    "estimated_alpha": alpha,
                }
            )
            print(f"{venue:7} {month}: alpha={alpha:.4f}, sentences={sentence_count}")

    write_csv(rows, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
