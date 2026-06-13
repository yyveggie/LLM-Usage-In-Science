#!/usr/bin/env python3
"""Pluggable LLM access + the paper's two-stage AI-text generation.

The original papers build the AI-modified distribution Q by generating
LLM-produced scientific text with a *single* model (gpt-3.5-turbo). This module
makes that generation model pluggable: every model is reached through an
OpenAI-compatible Chat Completions API, configured in ``models.yaml``.

Only the generation model changes. The prompts (two-stage summarize -> expand,
plus an optional proofread variant) are reproduced from Appendix C of
"Mapping the Increasing Use of LLMs in Scientific Papers".

This module is import-safe without the ``openai`` package installed; the client
is only constructed when you actually generate text.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.paths import CONFIG_DIR


DEFAULT_CONFIG = CONFIG_DIR / "models.yaml"


# --- Prompts reproduced from the paper (Appendix C, Figures 9-11) ------------

SUMMARIZE_SYSTEM = (
    "The aim here is to reverse-engineer the author's writing process by taking "
    "a piece of text from a paper and compressing it into a more concise form. "
    "This process simulates how an author might distill their thoughts and key "
    "points into a structured, yet not overly condensed form. Now as a first "
    "step, first summarize the goal of the text, e.g., is it introduction, or "
    "method, results? and then given a complete piece of text from a paper, "
    "reverse-engineer it into a list of bullet points."
)

EXPAND_SYSTEM = (
    "Following the initial step of reverse-engineering the author's writing "
    "process by compressing a text segment from a paper, you now enter the "
    "second phase. Here, your objective is to expand upon the concise version "
    "previously crafted. This stage simulates how an author elaborates on the "
    "distilled thoughts and key points, enriching them into a detailed, "
    "structured narrative. Given the concise output from the previous step, "
    "your task is to develop it into a fully fleshed-out text."
)

PROOFREAD_SYSTEM = (
    "Your task is to proofread the provided sentence for grammatical accuracy. "
    "Ensure that the corrections introduce minimal distortion to the original "
    "content."
)


@dataclass(frozen=True)
class ModelConfig:
    """Resolved configuration for one generation model."""

    name: str
    base_url: str
    model: str
    api_key_env: str
    params: dict[str, Any] = field(default_factory=dict)
    request_timeout: float = 120.0
    max_retries: int = 5
    retry_backoff: float = 2.0
    drop_params: tuple[str, ...] = ()

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env!r} is empty. "
                f"Export it before generating with model {self.name!r}."
            )
        return key


_GENERATION_KEYS = (
    "temperature",
    "top_p",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
)


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, ModelConfig]:
    """Parse ``models.yaml`` into a mapping of model name -> ModelConfig."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    defaults: dict[str, Any] = dict(raw.get("defaults", {}))
    models_raw: dict[str, Any] = raw.get("models", {})
    if not models_raw:
        raise ValueError(f"No models defined in {path}")

    configs: dict[str, ModelConfig] = {}
    for name, entry in models_raw.items():
        entry = dict(entry or {})
        merged = {**defaults, **entry}
        drop = tuple(merged.get("drop_params", []) or [])

        params = {
            key: merged[key]
            for key in _GENERATION_KEYS
            if key in merged and key not in drop
        }

        configs[name] = ModelConfig(
            name=name,
            base_url=str(merged["base_url"]),
            model=str(merged["model"]),
            api_key_env=str(merged["api_key_env"]),
            params=params,
            request_timeout=float(merged.get("request_timeout", 120.0)),
            max_retries=int(merged.get("max_retries", 5)),
            retry_backoff=float(merged.get("retry_backoff", 2.0)),
            drop_params=drop,
        )
    return configs


def get_model_config(name: str, path: Path | str = DEFAULT_CONFIG) -> ModelConfig:
    configs = load_config(path)
    if name not in configs:
        available = ", ".join(sorted(configs))
        raise KeyError(f"Model {name!r} not in config. Available: {available}")
    return configs[name]


class LLMClient:
    """Thin wrapper over an OpenAI-compatible Chat Completions endpoint."""

    def __init__(self, config: ModelConfig):
        self.config = config
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "The 'openai' package is required to generate text. "
                "Install it with: pip install -r requirements-llm-swap.txt"
            ) from exc

        self._client = OpenAI(
            api_key=config.api_key(),
            base_url=config.base_url,
            timeout=config.request_timeout,
        )

    def complete(self, system: str, user: str) -> str:
        """Single chat completion with retry/backoff. Returns the message text."""
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    **self.config.params,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001 - provider-agnostic retry
                last_error = exc
                if attempt + 1 >= self.config.max_retries:
                    break
                time.sleep(self.config.retry_backoff ** attempt)
        raise RuntimeError(
            f"Model {self.config.name!r} failed after "
            f"{self.config.max_retries} attempts: {last_error}"
        )

    # --- Paper's generation strategies -----------------------------------

    def two_stage(self, human_paragraph: str) -> str:
        """Counterfactual generation: summarize to an outline, then expand.

        Mirrors the paper's main strategy for building realistic LLM-produced
        scientific text without fabricating results.
        """
        outline = self.complete(SUMMARIZE_SYSTEM, human_paragraph)
        return self.complete(EXPAND_SYSTEM, outline)

    def proofread(self, human_text: str) -> str:
        """Minimal-edit proofreading variant (robustness check in the paper)."""
        return self.complete(PROOFREAD_SYSTEM, human_text)

    def generate(self, human_paragraph: str, strategy: str = "two_stage") -> str:
        if strategy == "two_stage":
            return self.two_stage(human_paragraph)
        if strategy == "proofread":
            return self.proofread(human_paragraph)
        raise ValueError(f"Unknown generation strategy: {strategy!r}")


def list_models(path: Path | str = DEFAULT_CONFIG) -> None:
    """Print configured models (handy CLI sanity check)."""
    for name, cfg in load_config(path).items():
        print(f"{name:16} -> {cfg.model} @ {cfg.base_url} (key: {cfg.api_key_env})")


if __name__ == "__main__":
    list_models()
