#!/usr/bin/env python3
"""Pluggable tokenizers, so generated AI text matches the official vocabulary.

The official datasets (Liang et al.) were tokenized with spaCy's small English
model. When we rebuild Q for a new LLM, the AI corpus must be tokenized the same
way as the official human/inference data; otherwise the same surface word counts
differently and q_t is biased, which in turn biases alpha.

This module exposes two tokenizers behind one interface:

- ``regex``  : the lightweight stdlib tokenizer used by the synthetic demo.
- ``spacy``  : spaCy ``en_core_web_sm`` (recommended; matches the papers).

Use ``inspect_official_vocab.py`` to measure how well a tokenizer's output
overlaps the official ``Word`` vocabulary before trusting the alpha estimates.
"""

from __future__ import annotations

from typing import Callable

from core.distributional_llm_reproduction import tokenize as regex_tokenize


Tokenizer = Callable[[str], list[str]]

_SPACY_NLP = None  # lazily loaded spaCy pipeline


def spacy_tokenize(text: str) -> list[str]:
    """Lowercased alphabetic tokens via spaCy ``en_core_web_sm``.

    Mirrors the official preprocessing: spaCy tokenization, lowercased, keeping
    word tokens (``is_alpha``) and dropping punctuation/numbers/whitespace.
    """
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "spaCy is required for the 'spacy' tokenizer. Install with:\n"
                "  pip install spacy && python -m spacy download en_core_web_sm"
            ) from exc
        try:
            _SPACY_NLP = spacy.load(
                "en_core_web_sm",
                disable=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"],
            )
        except OSError as exc:  # pragma: no cover - model not downloaded
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. Download it with:\n"
                "  python -m spacy download en_core_web_sm"
            ) from exc

    return [token.text.lower() for token in _SPACY_NLP(text) if token.is_alpha]


_TOKENIZERS: dict[str, Tokenizer] = {
    "regex": regex_tokenize,
    "spacy": spacy_tokenize,
}

TOKENIZER_CHOICES = tuple(_TOKENIZERS)


def get_tokenizer(name: str) -> Tokenizer:
    if name not in _TOKENIZERS:
        raise ValueError(
            f"Unknown tokenizer {name!r}. Choose from: {', '.join(TOKENIZER_CHOICES)}"
        )
    return _TOKENIZERS[name]
