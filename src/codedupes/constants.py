"""Shared package-level defaults used across CLI and analysis modules."""

from __future__ import annotations

from typing import Literal, cast

SemanticTask = Literal[
    "semantic-similarity",
    "code-retrieval",
    "retrieval",
    "question-answering",
    "fact-verification",
    "classification",
    "clustering",
]

DEFAULT_MODEL = "gte-modernbert-base"
DEFAULT_SEMANTIC_THRESHOLD = 0.82
DEFAULT_TRADITIONAL_THRESHOLD = 0.85
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEMANTIC_DEVICE = "auto"
SEMANTIC_DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
DEFAULT_MIN_SEMANTIC_LINES = 3
DEFAULT_TOP_K = 10
DEFAULT_CHECK_SEMANTIC_TASK: SemanticTask = "semantic-similarity"
DEFAULT_SEARCH_SEMANTIC_TASK: SemanticTask = "code-retrieval"
SEMANTIC_TASK_CHOICES: tuple[SemanticTask, ...] = (
    "semantic-similarity",
    "code-retrieval",
    "retrieval",
    "question-answering",
    "fact-verification",
    "classification",
    "clustering",
)


def normalize_semantic_task(
    semantic_task: str | None,
    *,
    default_task: SemanticTask,
) -> SemanticTask:
    """Validate and normalize one semantic task name.

    :param semantic_task: Candidate task value.
    :param default_task: Fallback task when no value is provided.
    :return: Normalized task name.
    :raises ValueError: If the task is not supported.
    """
    if semantic_task is None:
        return default_task

    normalized = semantic_task.strip().lower()
    if normalized not in SEMANTIC_TASK_CHOICES:
        allowed = ", ".join(SEMANTIC_TASK_CHOICES)
        raise ValueError(f"Invalid semantic_task: {semantic_task}. Allowed values: {allowed}")
    return cast(SemanticTask, normalized)
