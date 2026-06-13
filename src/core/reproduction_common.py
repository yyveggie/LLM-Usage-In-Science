#!/usr/bin/env python3
"""Shared helpers for the three paper-specific reproduction entry points."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from core.distributional_llm_reproduction import (
    DistributionModel,
    estimate_alpha,
    precompute_log_probabilities,
)
from core.paths import OFFICIAL_DATA, RESULTS

VENUE_CONFIG = {
    "CS": {
        "distribution": OFFICIAL_DATA / "distribution" / "CS.parquet",
        "inference_dir": OFFICIAL_DATA / "inference_data" / "arxiv" / "CS",
    },
    "EESS": {
        "distribution": OFFICIAL_DATA / "distribution" / "EESS.parquet",
        "inference_dir": OFFICIAL_DATA / "inference_data" / "arxiv" / "EESS",
    },
    "Math": {
        "distribution": OFFICIAL_DATA / "distribution" / "Math.parquet",
        "inference_dir": OFFICIAL_DATA / "inference_data" / "arxiv" / "Math",
    },
    "Phys": {
        "distribution": OFFICIAL_DATA / "distribution" / "Phys.parquet",
        "inference_dir": OFFICIAL_DATA / "inference_data" / "arxiv" / "Phys",
    },
    "Stat": {
        "distribution": OFFICIAL_DATA / "distribution" / "Stat.parquet",
        "inference_dir": OFFICIAL_DATA / "inference_data" / "arxiv" / "Stat",
    },
    "bioRxiv": {
        "distribution": OFFICIAL_DATA / "distribution" / "Biorxiv.parquet",
        "inference_dir": OFFICIAL_DATA / "inference_data" / "biorxiv",
    },
    "Nature": {
        "distribution": OFFICIAL_DATA / "distribution" / "Nature.parquet",
        "inference_dir": OFFICIAL_DATA / "inference_data" / "nature",
    },
}


def load_official_distribution(path: Path) -> DistributionModel:
    """Load a paper-provided distribution parquet file."""
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


def _as_token_list(value: Any) -> list[str]:
    return [str(token) for token in list(value)]


def load_tokenized_sentences(path: Path) -> list[list[str]]:
    """Load tokenized sentences from official inference/validation parquet.

    Official validation files store one tokenized sentence per row.
    Official monthly inference files store one abstract per row, where the
    value is an array/list of tokenized sentences. This loader supports both.
    """
    df = pd.read_parquet(path)
    if "inference_sentence" not in df.columns:
        raise ValueError(f"{path} has no inference_sentence column")

    sentences: list[list[str]] = []
    for value in df["inference_sentence"]:
        if value is None:
            continue
        items = list(value)
        if not items:
            continue
        first = items[0]
        if isinstance(first, str):
            tokens = _as_token_list(items)
            if len(tokens) > 1:
                sentences.append(tokens)
        else:
            for sentence in items:
                tokens = _as_token_list(sentence)
                if len(tokens) > 1:
                    sentences.append(tokens)
    return sentences


def estimate_alpha_for_file(model: DistributionModel, path: Path) -> tuple[float, int]:
    sentences = load_tokenized_sentences(path)
    log_p_values, log_q_values = precompute_log_probabilities(sentences, model)
    return estimate_alpha(log_p_values, log_q_values), len(sentences)


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
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


def month_file(venue: str, month: str) -> Path:
    config = VENUE_CONFIG[venue]
    return config["inference_dir"] / f"{month}.parquet"


def distribution_file(venue: str) -> Path:
    return VENUE_CONFIG[venue]["distribution"]
