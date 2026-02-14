"""Shared package-level defaults used across CLI and analysis modules."""

from __future__ import annotations

from codedupes.semantic_profiles import DEFAULT_C2LLM_REVISION as PROFILE_DEFAULT_C2LLM_REVISION

DEFAULT_MODEL = "gte-modernbert-base"
DEFAULT_C2LLM_REVISION = PROFILE_DEFAULT_C2LLM_REVISION
DEFAULT_SEMANTIC_THRESHOLD = 0.82
DEFAULT_TRADITIONAL_THRESHOLD = 0.85
DEFAULT_BATCH_SIZE = 8
DEFAULT_MIN_SEMANTIC_LINES = 3
DEFAULT_TOP_K = 10
DEFAULT_CHECK_SEMANTIC_TASK = "semantic-similarity"
DEFAULT_SEARCH_SEMANTIC_TASK = "code-retrieval"
SEMANTIC_TASK_CHOICES = (
    "semantic-similarity",
    "code-retrieval",
    "retrieval",
    "question-answering",
    "fact-verification",
    "classification",
    "clustering",
)
