#!/usr/bin/env python3
"""Core-method entry for:

AI-Assisted Writing Is Growing Fastest Among Non-English-Speaking and
Less Established Scientists.

What this reproduces:
- The shared distributional LLM-use estimator on a known-alpha toy corpus.
- The paper-specific DiD/DDD regression logic on a transparent demo panel.

What this does not reproduce:
- The unavailable full PubMed Central + OpenAlex + QS data pipeline.
"""

from __future__ import annotations

# Allow running directly from the project root without installing the package.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))

import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.distributional_llm_reproduction import run_synthetic_validation
from core.reproduction_common import RESULTS, write_csv


def ols(y: np.ndarray, x: np.ndarray, names: list[str]) -> list[dict[str, float | str]]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    dof = max(1, x.shape[0] - x.shape[1])
    sigma2 = float(residuals @ residuals / dof)
    covariance = sigma2 * np.linalg.inv(x.T @ x)
    standard_errors = np.sqrt(np.diag(covariance))
    return [
        {
            "term": name,
            "coefficient": float(coef),
            "standard_error": float(se),
        }
        for name, coef, se in zip(names, beta, standard_errors, strict=True)
    ]


def build_demo_panel(n: int, seed: int) -> list[dict[str, float | int]]:
    rng = random.Random(seed)
    rows: list[dict[str, float | int]] = []
    for _ in range(n):
        non_english = int(rng.random() < 0.55)
        post_gpt = int(rng.random() < 0.50)
        less_established = int(rng.random() < 0.60)
        ai_experience = int(rng.random() < 0.20)

        llm_use = (
            0.045
            + 0.010 * non_english
            + 0.030 * post_gpt
            + 0.030 * non_english * post_gpt
            + 0.016 * less_established * post_gpt
            + 0.012 * non_english * less_established * post_gpt
            + 0.014 * ai_experience * post_gpt
            + rng.gauss(0.0, 0.012)
        )
        rows.append(
            {
                "llm_use": max(0.0, min(1.0, llm_use)),
                "non_english": non_english,
                "post_gpt": post_gpt,
                "less_established": less_established,
                "ai_experience": ai_experience,
            }
        )
    return rows


def run_did_ddd_demo(n: int, seed: int) -> list[dict[str, float | str]]:
    panel = build_demo_panel(n, seed)
    y = np.array([row["llm_use"] for row in panel], dtype=float)
    non_english = np.array([row["non_english"] for row in panel], dtype=float)
    post = np.array([row["post_gpt"] for row in panel], dtype=float)
    less = np.array([row["less_established"] for row in panel], dtype=float)
    ai_exp = np.array([row["ai_experience"] for row in panel], dtype=float)

    did_x = np.column_stack(
        [
            np.ones(len(panel)),
            non_english,
            post,
            non_english * post,
            ai_exp * post,
        ]
    )
    did_rows = ols(
        y,
        did_x,
        ["intercept", "non_english", "post_gpt", "non_english:post_gpt", "ai_experience:post_gpt"],
    )
    for row in did_rows:
        row["model"] = "DiD"

    ddd_x = np.column_stack(
        [
            np.ones(len(panel)),
            non_english,
            post,
            less,
            non_english * post,
            less * post,
            non_english * less,
            non_english * less * post,
            ai_exp * post,
        ]
    )
    ddd_rows = ols(
        y,
        ddd_x,
        [
            "intercept",
            "non_english",
            "post_gpt",
            "less_established",
            "non_english:post_gpt",
            "less_established:post_gpt",
            "non_english:less_established",
            "non_english:less_established:post_gpt",
            "ai_experience:post_gpt",
        ],
    )
    for row in ddd_rows:
        row["model"] = "DDD"

    return did_rows + ddd_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper 1 core-method reproduction.")
    parser.add_argument("--panel-size", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--output",
        default=RESULTS / "01_ai_assisted_non_english_core.csv",
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_rows = run_synthetic_validation(
        SimpleNamespace(
            train_size=1200,
            inference_size=1000,
            bootstrap=40,
            min_occurrences=2,
            seed=args.seed,
            alphas=[0.0, 0.10, 0.20],
        )
    )
    regression_rows = run_did_ddd_demo(args.panel_size, args.seed)

    output_rows: list[dict[str, float | str]] = []
    for row in validation_rows:
        output_rows.append(
            {
                "paper": "AI-Assisted Writing",
                "section": "distributional_validation",
                "model": "known_alpha",
                "term": f"alpha={row['ground_truth_alpha']:.3f}",
                "coefficient": row["estimated_alpha"],
                "standard_error": row["absolute_error"],
            }
        )
    for row in regression_rows:
        output_rows.append(
            {
                "paper": "AI-Assisted Writing",
                "section": "did_ddd_demo",
                **row,
            }
        )

    write_csv(output_rows, args.output)
    key_terms = {
        row["term"]: row["coefficient"]
        for row in regression_rows
        if row["term"] in {"non_english:post_gpt", "non_english:less_established:post_gpt"}
    }
    print("Paper 1 core method complete.")
    print(f"DiD non_english:post_gpt coefficient: {key_terms['non_english:post_gpt']:.4f}")
    print(
        "DDD non_english:less_established:post_gpt coefficient: "
        f"{key_terms['non_english:less_established:post_gpt']:.4f}"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
