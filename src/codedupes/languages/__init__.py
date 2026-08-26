"""Polyglot extraction support for codedupes."""

from codedupes.languages.registry import (
    GRAMMAR_PACKAGES,
    REQUIRED_PARSER_PACKAGES,
    SUPPORTED_LANGUAGES,
    GrammarStatus,
    LanguageSelection,
    get_backend,
    get_grammar_statuses,
    language_for_path,
    normalize_languages,
    repository_allows_c_headers,
)
from codedupes.languages.tree_sitter_backend import GrammarUnavailableError

__all__ = [
    "GRAMMAR_PACKAGES",
    "REQUIRED_PARSER_PACKAGES",
    "SUPPORTED_LANGUAGES",
    "GrammarStatus",
    "GrammarUnavailableError",
    "LanguageSelection",
    "get_backend",
    "get_grammar_statuses",
    "language_for_path",
    "normalize_languages",
    "repository_allows_c_headers",
]
