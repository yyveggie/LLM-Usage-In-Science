#!/usr/bin/env python3
"""Run validation on the official packaged CS data.

This script uses files downloaded from:
https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers

Inputs:
- official_data/distribution/CS.parquet
- official_data/validation_data/CS/ground_truth_alpha_*.parquet

The official validation parquet files contain tokenized sentences with known
ground-truth alpha values. This script estimates alpha from the official
distribution file and writes a CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

from core.distributional_llm_reproduction import (
    DistributionModel,
    bootstrap_alpha_ci,
    estimate_alpha,
    precompute_log_probabilities,
)
from core.paths import OFFICIAL_DATA, RESULTS


ALPHA_RE = re.compile(r"ground_truth_alpha_(.+)\.parquet$")


def load_official_distribution(path: Path) -> DistributionModel:
    df = pd.read_parquet(path)
    required = {"Word", "logP", "logQ", "log1-P", "log1-Q"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    words = df["Word"].astype(str).tolist()
    return DistributionModel(
        vocabulary=tuple(words),
        log_p=dict(zip(words, df["logP"].astype(float), strict=True)),
        log_q=dict(zip(words, df["logQ"].astype(float), strict=True)),
        log_not_p=dict(zip(words, df["log1-P"].astype(float), strict=True)),
        log_not_q=dict(zip(words, df["log1-Q"].astype(float), strict=True)),
    )


def alpha_from_path(path: Path) -> float:
    match = ALPHA_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer alpha from file name: {path}")
    return float(match.group(1))


def load_sentences(path: Path) -> list[list[str]]:
    df = pd.read_parquet(path)
    if "inference_sentence" not in df.columns:
        raise ValueError(f"{path} has no inference_sentence column")
    return [list(sentence) for sentence in df["inference_sentence"] if len(sentence) > 1]


def run_validation(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    model = load_official_distribution(args.distribution)
    files = sorted(args.validation_dir.glob("ground_truth_alpha_*.parquet"), key=alpha_from_path)
    if not files:
        raise ValueError(f"No validation parquet files found in {args.validation_dir}")

    rows: list[dict[str, float | int | str]] = []
    for path in files:
        ground_truth = alpha_from_path(path)
        sentences = load_sentences(path)
        log_p_values, log_q_values = precompute_log_probabilities(sentences, model)
        estimated = estimate_alpha(log_p_values, log_q_values)

        row: dict[str, float | int | str] = {
            "file": path.name,
            "sentence_count": len(sentences),
            "ground_truth_alpha": ground_truth,
            "estimated_alpha": estimated,
            "absolute_error": abs(estimated - ground_truth),
        }

        if args.bootstrap > 0:
            ci_low, ci_high = bootstrap_alpha_ci(
                log_p_values,
                log_q_values,
                n_bootstrap=args.bootstrap,
                seed=args.seed + int(ground_truth * 1000),
            )
            row["ci_low"] = ci_low
            row["ci_high"] = ci_high

        rows.append(row)
        print(
            f"{path.name}: ground_truth={ground_truth:.3f}, "
            f"estimated={estimated:.3f}, error={abs(estimated - ground_truth):.3f}"
        )

    return rows


def write_rows(rows: list[dict[str, float | int | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate alpha estimates on official CS data.")
    parser.add_argument(
        "--distribution",
        type=Path,
        default=OFFICIAL_DATA / "distribution" / "CS.parquet",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=OFFICIAL_DATA / "validation_data" / "CS",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "official_cs_validation.csv",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Optional bootstrap repetitions. Set to 0 for fast point estimates.",
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_validation(args)
    write_rows(rows, args.output)
    max_error = max(float(row["absolute_error"]) for row in rows)
    print(f"\nSaved results to {args.output}")
    print(f"Maximum absolute error: {max_error:.3f}")


if __name__ == "__main__":
    main()
