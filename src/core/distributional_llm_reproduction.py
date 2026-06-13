#!/usr/bin/env python3
"""Minimal reproduction of distributional LLM quantification.

This script reproduces the core method used by Liang et al.:

1. Estimate token-occurrence distributions from human and AI corpora.
2. Treat a target corpus as a mixture of those two distributions.
3. Estimate the mixture weight alpha by maximum likelihood.

It is intentionally lightweight and uses only the Python standard library.
The included demo uses synthetic corpora with known alpha values, so the
method can be verified without downloading the full paper datasets.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.paths import RESULTS


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


COMMON_WORDS = [
    "analysis",
    "approach",
    "authors",
    "data",
    "evidence",
    "experiment",
    "finding",
    "method",
    "model",
    "paper",
    "research",
    "result",
    "sample",
    "science",
    "study",
    "system",
]

HUMAN_STYLE_WORDS = [
    "also",
    "because",
    "case",
    "compare",
    "describe",
    "estimate",
    "however",
    "measure",
    "report",
    "show",
    "specific",
    "test",
    "use",
    "variation",
]

AI_STYLE_WORDS = [
    "commendable",
    "comprehensive",
    "delve",
    "enhance",
    "intricate",
    "leverage",
    "notably",
    "pivotal",
    "realm",
    "robust",
    "seamless",
    "showcase",
    "underscore",
    "utilize",
]


@dataclass(frozen=True)
class DistributionModel:
    vocabulary: tuple[str, ...]
    log_p: dict[str, float]
    log_q: dict[str, float]
    log_not_p: dict[str, float]
    log_not_q: dict[str, float]


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for demonstration purposes."""
    return [match.group(0).lower().strip("'") for match in TOKEN_RE.finditer(text)]


def count_binary_occurrences(sentences: Iterable[Iterable[str]]) -> Counter[str]:
    """Count in how many sentences each token appears."""
    counts: Counter[str] = Counter()
    for sentence in sentences:
        counts.update(set(sentence))
    return counts


def fit_distribution_model(
    human_sentences: list[list[str]],
    ai_sentences: list[list[str]],
    *,
    min_human_occurrences: int = 2,
    min_ai_occurrences: int = 2,
    smoothing: float = 0.5,
) -> DistributionModel:
    """Estimate Bernoulli token-occurrence distributions P and Q.

    P is the human-written distribution, Q is the AI-modified distribution.
    The official code filters to frequent tokens appearing in both corpora.
    This reproduction follows the same idea and adds light smoothing to avoid
    infinite log probabilities in small toy corpora.
    """
    if not human_sentences or not ai_sentences:
        raise ValueError("Both human_sentences and ai_sentences must be non-empty.")

    human_counts = count_binary_occurrences(human_sentences)
    ai_counts = count_binary_occurrences(ai_sentences)
    common_vocab = set(human_counts).intersection(ai_counts)
    vocabulary = tuple(
        sorted(
            token
            for token in common_vocab
            if human_counts[token] >= min_human_occurrences
            and ai_counts[token] >= min_ai_occurrences
        )
    )
    if not vocabulary:
        raise ValueError("No shared vocabulary survived the frequency filters.")

    n_human = len(human_sentences)
    n_ai = len(ai_sentences)
    denominator_human = n_human + 2.0 * smoothing
    denominator_ai = n_ai + 2.0 * smoothing

    log_p: dict[str, float] = {}
    log_q: dict[str, float] = {}
    log_not_p: dict[str, float] = {}
    log_not_q: dict[str, float] = {}

    for token in vocabulary:
        p = (human_counts[token] + smoothing) / denominator_human
        q = (ai_counts[token] + smoothing) / denominator_ai
        log_p[token] = math.log(p)
        log_q[token] = math.log(q)
        log_not_p[token] = math.log1p(-p)
        log_not_q[token] = math.log1p(-q)

    return DistributionModel(vocabulary, log_p, log_q, log_not_p, log_not_q)


def precompute_log_probabilities(
    sentences: list[list[str]], model: DistributionModel
) -> tuple[list[float], list[float]]:
    """Compute log P_T(x) and log Q_T(x) for every sentence."""
    vocab = set(model.vocabulary)
    base_log_p = sum(model.log_not_p.values())
    base_log_q = sum(model.log_not_q.values())
    log_p_values: list[float] = []
    log_q_values: list[float] = []

    for sentence in sentences:
        present_tokens = set(sentence).intersection(vocab)
        log_p = base_log_p
        log_q = base_log_q
        for token in present_tokens:
            log_p += model.log_p[token] - model.log_not_p[token]
            log_q += model.log_q[token] - model.log_not_q[token]
        log_p_values.append(log_p)
        log_q_values.append(log_q)

    return log_p_values, log_q_values


def logaddexp(a: float, b: float) -> float:
    """Stable log(exp(a) + exp(b))."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    high = max(a, b)
    return high + math.log(math.exp(a - high) + math.exp(b - high))


def negative_log_likelihood(
    alpha: float, log_p_values: list[float], log_q_values: list[float]
) -> float:
    """Negative mean log likelihood under (1-alpha)P + alpha Q."""
    if not 0.0 <= alpha <= 1.0:
        return math.inf

    if alpha == 0.0:
        return -sum(log_p_values) / len(log_p_values)
    if alpha == 1.0:
        return -sum(log_q_values) / len(log_q_values)

    log_human_weight = math.log1p(-alpha)
    log_ai_weight = math.log(alpha)
    total = 0.0
    for log_p, log_q in zip(log_p_values, log_q_values, strict=True):
        total += logaddexp(log_human_weight + log_p, log_ai_weight + log_q)
    return -total / len(log_p_values)


def estimate_alpha(
    log_p_values: list[float], log_q_values: list[float], *, iterations: int = 80
) -> float:
    """One-dimensional MLE for alpha using golden-section search."""
    low = 0.0
    high = 1.0
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    inv_phi_sq = (3.0 - math.sqrt(5.0)) / 2.0

    c = low + inv_phi_sq * (high - low)
    d = low + inv_phi * (high - low)
    f_c = negative_log_likelihood(c, log_p_values, log_q_values)
    f_d = negative_log_likelihood(d, log_p_values, log_q_values)

    for _ in range(iterations):
        if f_c < f_d:
            high = d
            d = c
            f_d = f_c
            c = low + inv_phi_sq * (high - low)
            f_c = negative_log_likelihood(c, log_p_values, log_q_values)
        else:
            low = c
            c = d
            f_c = f_d
            d = low + inv_phi * (high - low)
            f_d = negative_log_likelihood(d, log_p_values, log_q_values)

    interior_alpha = (low + high) / 2.0
    candidates = [
        (0.0, negative_log_likelihood(0.0, log_p_values, log_q_values)),
        (interior_alpha, negative_log_likelihood(interior_alpha, log_p_values, log_q_values)),
        (1.0, negative_log_likelihood(1.0, log_p_values, log_q_values)),
    ]
    return min(candidates, key=lambda item: item[1])[0]


def bootstrap_alpha_ci(
    log_p_values: list[float],
    log_q_values: list[float],
    *,
    n_bootstrap: int = 120,
    seed: int = 7,
) -> tuple[float, float]:
    """Bootstrap a simple percentile confidence interval for alpha."""
    rng = random.Random(seed)
    n = len(log_p_values)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        sample_log_p: list[float] = []
        sample_log_q: list[float] = []
        for _ in range(n):
            index = rng.randrange(n)
            sample_log_p.append(log_p_values[index])
            sample_log_q.append(log_q_values[index])
        estimates.append(estimate_alpha(sample_log_p, sample_log_q))

    estimates.sort()
    low_index = max(0, math.floor(0.025 * (len(estimates) - 1)))
    high_index = min(len(estimates) - 1, math.ceil(0.975 * (len(estimates) - 1)))
    return estimates[low_index], estimates[high_index]


def sample_sentence(kind: str, rng: random.Random) -> list[str]:
    """Generate one toy academic sentence as tokens."""
    if kind not in {"human", "ai"}:
        raise ValueError("kind must be 'human' or 'ai'.")

    if kind == "human":
        primary = HUMAN_STYLE_WORDS
        secondary = AI_STYLE_WORDS
    else:
        primary = AI_STYLE_WORDS
        secondary = HUMAN_STYLE_WORDS

    tokens = []
    tokens.extend(rng.choices(COMMON_WORDS, k=7))
    tokens.extend(rng.choices(primary, k=4))
    if rng.random() < 0.35:
        tokens.extend(rng.choices(secondary, k=1))
    tokens.extend(rng.choices(COMMON_WORDS + primary + secondary, k=2))
    rng.shuffle(tokens)
    return tokens


def make_corpus(kind: str, size: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    return [sample_sentence(kind, rng) for _ in range(size)]


def make_mixed_corpus(
    human_pool: list[list[str]],
    ai_pool: list[list[str]],
    *,
    alpha: float,
    size: int,
    seed: int,
) -> list[list[str]]:
    """Create a target corpus with a known fraction alpha of AI sentences."""
    rng = random.Random(seed)
    ai_count = round(alpha * size)
    human_count = size - ai_count
    mixed = rng.choices(human_pool, k=human_count) + rng.choices(ai_pool, k=ai_count)
    rng.shuffle(mixed)
    return mixed


def run_synthetic_validation(args: argparse.Namespace) -> list[dict[str, float]]:
    human_train = make_corpus("human", args.train_size, seed=args.seed)
    ai_train = make_corpus("ai", args.train_size, seed=args.seed + 1)
    model = fit_distribution_model(
        human_train,
        ai_train,
        min_human_occurrences=args.min_occurrences,
        min_ai_occurrences=args.min_occurrences,
    )

    human_pool = make_corpus("human", args.inference_size * 2, seed=args.seed + 2)
    ai_pool = make_corpus("ai", args.inference_size * 2, seed=args.seed + 3)

    rows: list[dict[str, float]] = []
    for index, alpha in enumerate(args.alphas):
        mixed = make_mixed_corpus(
            human_pool,
            ai_pool,
            alpha=alpha,
            size=args.inference_size,
            seed=args.seed + 100 + index,
        )
        log_p_values, log_q_values = precompute_log_probabilities(mixed, model)
        estimate = estimate_alpha(log_p_values, log_q_values)
        ci_low, ci_high = bootstrap_alpha_ci(
            log_p_values,
            log_q_values,
            n_bootstrap=args.bootstrap,
            seed=args.seed + 200 + index,
        )
        rows.append(
            {
                "ground_truth_alpha": alpha,
                "estimated_alpha": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "absolute_error": abs(estimate - alpha),
            }
        )

    return rows


def write_results(rows: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_results(rows: list[dict[str, float]]) -> None:
    print("ground_truth_alpha, estimated_alpha, ci_low, ci_high, absolute_error")
    for row in rows:
        print(
            f"{row['ground_truth_alpha']:.3f}, "
            f"{row['estimated_alpha']:.3f}, "
            f"{row['ci_low']:.3f}, "
            f"{row['ci_high']:.3f}, "
            f"{row['absolute_error']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce distributional LLM quantification on a toy corpus."
    )
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--inference-size", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=120)
    parser.add_argument("--min-occurrences", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
        help="Known mixture weights to validate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "synthetic_validation.csv",
    )
    parser.add_argument(
        "--check-tolerance",
        type=float,
        default=0.04,
        help="Exit with an error if any absolute error exceeds this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_synthetic_validation(args)
    write_results(rows, args.output)
    print_results(rows)
    print(f"\nSaved results to {args.output}")

    max_error = max(row["absolute_error"] for row in rows)
    if max_error > args.check_tolerance:
        raise SystemExit(
            f"Maximum error {max_error:.3f} exceeds tolerance {args.check_tolerance:.3f}."
        )


if __name__ == "__main__":
    main()
