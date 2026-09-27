"""Shared builders and runners for the test_analyzer test modules."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np

import codedupes.semantic as semantic_module
from codedupes import analyzer as analyzer_module
from codedupes.models import CodeUnit, DuplicatePair

# Derived from the signature actually called at the analyzer's call site
# (analyzer_module.run_semantic_analysis is run_semantic_analysis_with_identity)
# so an added/removed parameter fails loudly here instead of drifting silently.
SEMANTIC_ANALYSIS_KWARG_NAMES = frozenset(
    inspect.signature(analyzer_module.run_semantic_analysis).parameters
) - {"units"}


def embedding_identity_from_kwargs(kwargs: dict[str, object]):
    """Build the effective test identity for forwarded semantic arguments."""
    return semantic_module.resolve_embedding_space_identity(
        model_name=str(kwargs.get("model_name", analyzer_module.DEFAULT_MODEL)),
        instruction_prefix=kwargs.get("instruction_prefix"),
        revision=kwargs.get("revision"),
        trust_remote_code=kwargs.get("trust_remote_code"),
        semantic_task=kwargs.get("semantic_task"),
        device=str(kwargs.get("device", "cpu")),
        mps_fallback=kwargs.get("mps_fallback"),
        persist_local_model_manifest=False,
        strict_revision_cache=bool(kwargs.get("strict_revision_cache", False)),
    )


def make_semantic_runner(
    *,
    duplicate_factory: Callable[[list[CodeUnit]], list[DuplicatePair]] | None = None,
    capture: dict[str, object] | None = None,
    capture_exclude_pairs: set[tuple[str, str]] | None = None,
    error: Exception | None = None,
) -> Callable[..., tuple[np.ndarray, list[DuplicatePair], object]]:
    """Build a reusable semantic-analysis test double."""

    def fake_run_semantic(units, **kwargs):
        assert set(kwargs) == SEMANTIC_ANALYSIS_KWARG_NAMES
        if capture is not None:
            capture.update(kwargs)
        if capture_exclude_pairs is not None:
            capture_exclude_pairs.update(kwargs["exclude_pairs"] or set())
        if error is not None:
            raise error

        duplicates = duplicate_factory(units) if duplicate_factory is not None else []
        return (
            np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(units), 1)),
            duplicates,
            embedding_identity_from_kwargs(kwargs),
        )

    return fake_run_semantic


def traditional_single_jaccard_runner(similarity: float = 0.9):
    """Build a traditional runner returning one jaccard duplicate for first two units."""

    def fake_traditional(units, jaccard_threshold=0.85):
        first, second = units[:2]
        return (
            [DuplicatePair(unit_a=first, unit_b=second, similarity=similarity, method="jaccard")],
            [],
        )

    return fake_traditional
