"""Polyglot extraction support for codedupes."""

from codedupes.languages.registry import (
    GRAMMAR_PACKAGES,
    SUPPORTED_LANGUAGES,
    GrammarStatus,
    LanguageSelection,
    get_grammar_statuses,
    language_for_path,
    normalize_languages,
)

__all__ = [
    "GRAMMAR_PACKAGES",
    "SUPPORTED_LANGUAGES",
    "GrammarStatus",
    "LanguageSelection",
    "get_grammar_statuses",
    "language_for_path",
    "normalize_languages",
]
