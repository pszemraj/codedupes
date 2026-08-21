"""Shared package-level defaults used across CLI and analysis modules."""

from __future__ import annotations

from typing import Literal, cast, get_args

SemanticTask = Literal[
    "semantic-similarity",
    "code-retrieval",
    "retrieval",
    "question-answering",
    "fact-verification",
    "classification",
    "clustering",
]

# Single source of truth: extraction skips these directories, and the C/C++ header
# probe prunes them so vendored sources cannot change a whole-tree decision.
DEFAULT_EXCLUDE_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".hypothesis",
        ".eggs",
        "build",
        "dist",
        "target",
        "node_modules",
        ".pnpm-store",
        ".yarn",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".gradle",
        ".idea",
        ".vscode",
        ".terraform",
        ".serverless",
        ".aws-sam",
        ".dart_tool",
    }
)

DEFAULT_MODEL = "gte-modernbert-base"
DEFAULT_TRADITIONAL_THRESHOLD = 0.85
DEFAULT_BATCH_SIZE = 8
# CPU OOM can arrive as an uncatchable OOM-killer SIGKILL rather than a Python
# exception (observed on WSL2), so the post-accelerator CPU retry must not restart
# at an arbitrarily large requested batch size.
CPU_FALLBACK_MAX_BATCH_SIZE = 32
DEFAULT_SEMANTIC_DEVICE = "auto"
SEMANTIC_DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
DEFAULT_MIN_SEMANTIC_STATEMENTS = 3
DEFAULT_TOP_K = 10
DEFAULT_CHECK_SEMANTIC_TASK: SemanticTask = "semantic-similarity"
DEFAULT_SEARCH_SEMANTIC_TASK: SemanticTask = "code-retrieval"
SEMANTIC_TASK_CHOICES: tuple[SemanticTask, ...] = get_args(SemanticTask)


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
