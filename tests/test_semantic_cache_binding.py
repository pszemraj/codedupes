"""Embedding-cache key binding: revision/trust identity, query-cache invariants, and CRLF normalization."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from codedupes import semantic
from codedupes.embedding_cache import EmbeddingCache, compute_cache_key
from codedupes.models import CodeUnit, CodeUnitType
from codedupes.semantic import (
    compute_embeddings,
    find_similar_to_query,
)
from tests.conftest import extract_arithmetic_units
from tests.semantic_helpers import FULL_REVISION, RecordingModel, WarmCacheModel


@pytest.mark.parametrize(
    ("cached_vector", "expected_encode_calls"),
    [
        (np.array([0.1, 0.0], dtype=np.float32), 0),
        (np.array([0.0, 0.0], dtype=np.float32), 1),
    ],
    ids=["renormalize", "zero-row-miss"],
)
def test_query_cache_hits_enforce_cosine_vector_invariants(
    tmp_path: Path,
    monkeypatch,
    cached_vector: np.ndarray,
    expected_encode_calls: int,
) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model = RecordingModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)
    profile = semantic.resolve_model_profile("gte-modernbert-base")
    plan = semantic.resolve_encode_plan("gte-modernbert-base", mode="query")
    cache, cache_revision, cache_variant, cache_namespace = semantic._prepare_cache_context(
        "query",
        profile,
        "gte-modernbert-base",
        FULL_REVISION,
        "cpu",
        plan,
        mps_fallback=None,
        trust_remote_code=False,
        use_cache=True,
        cache_scope=tmp_path,
    )
    assert cache is not None
    assert cache_revision is not None
    query = "find addition"
    cache_key = semantic.compute_cache_key(
        profile.canonical_name,
        cache_revision,
        semantic._prepare_embedding_text(query),
        mode="query",
        variant=cache_variant,
    )
    cache.put_many(
        tmp_path,
        profile.canonical_name,
        cache_revision,
        [(cache_key, cached_vector)],
        namespace=cache_namespace,
    )

    execution = []
    results = find_similar_to_query(
        query,
        units,
        embeddings,
        model_name="gte-modernbert-base",
        revision=FULL_REVISION,
        threshold=0.9,
        device="cpu",
        cache_scope=tmp_path,
        execution=execution,
    )

    assert results == [(units[0], 1.0)]
    assert len(model.encoded) == expected_encode_calls
    assert execution == [
        semantic.QueryExecution(
            execution_device="cpu" if expected_encode_calls else None,
            cache_hit=expected_encode_calls == 0,
            threshold=0.9,
        )
    ]


@pytest.mark.parametrize(
    ("revision", "expect_bypass"),
    [("main", True), ("b" * 40, False)],
)
def test_unreportable_mutable_provenance_bypasses_the_query_cache(
    monkeypatch, tmp_path: Path, revision: str, expect_bypass: bool
) -> None:
    # A corpus that had to bypass its shard (mutable branch, no reportable
    # commit) has no provenance to compare a cached query row against, so the
    # query cache must be bypassed with it.
    units = extract_arithmetic_units(tmp_path)
    model = RecordingModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    embeddings, identity = semantic.compute_embeddings_with_identity(
        units,
        model_name="test-model",
        revision=revision,
        cache_scope=tmp_path,
        strict_revision_cache=False,
    )
    assert identity.source_commit is None

    for _ in range(2):
        find_similar_to_query(
            "find addition",
            units,
            embeddings,
            model_name="test-model",
            revision=revision,
            cache_scope=tmp_path,
            strict_revision_cache=False,
            corpus_identity=identity,
            threshold=0.0,
        )

    query_encodes = [call for call in model.encoded if call == ["find addition"]]
    assert len(query_encodes) == (2 if expect_bypass else 1)


def test_query_embedding_cache_put_is_fifo_capped(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model = WarmCacheModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    captured: dict[str, object] = {}
    original_put_many = EmbeddingCache.put_many

    def _recording_put_many(self, *args, **kwargs):
        captured["max_namespace_keys"] = kwargs.get("max_namespace_keys")
        return original_put_many(self, *args, **kwargs)

    monkeypatch.setattr(EmbeddingCache, "put_many", _recording_put_many)

    find_similar_to_query(
        "find addition",
        units,
        embeddings,
        model_name="gte-modernbert-base",
        revision=FULL_REVISION,
        device="cpu",
        threshold=0.0,
        cache_scope=tmp_path,
    )

    assert captured["max_namespace_keys"] == semantic._MAX_CACHED_QUERY_KEYS


# --- T7: strict-by-default cache revision keying, loose opt-in -------------


@pytest.mark.parametrize(
    "function",
    [
        semantic.resolve_embedding_space_identity,
        semantic.compute_embeddings_with_identity,
        semantic.compute_embeddings,
        semantic.find_similar_to_query,
        semantic.run_semantic_analysis_with_identity,
        semantic.run_semantic_analysis,
    ],
)
def test_public_semantic_helpers_default_to_strict_revision_cache(function) -> None:
    parameter = inspect.signature(function).parameters["strict_revision_cache"]
    assert parameter.default is True


def test_resolve_revision_for_cache_loose_opt_in_labels_unpinned_model(monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("loose mode must never consult the offline hub-cache lookup")

    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", fail_if_called)

    assert semantic._resolve_revision_for_cache("some-generic-model", None, strict=False) == "main"
    assert (
        semantic._resolve_revision_for_cache("some-generic-model", "feature-branch", strict=False)
        == "feature-branch"
    )


def test_resolve_revision_for_cache_explicit_commit_hash_keys_as_is_either_mode() -> None:
    commit_hash = "a" * 40

    assert semantic._resolve_revision_for_cache("some-generic-model", commit_hash) == commit_hash
    assert (
        semantic._resolve_revision_for_cache("some-generic-model", commit_hash, strict=True)
        == commit_hash
    )


def test_resolve_revision_for_cache_strict_resolves_commit_and_disables_on_unmappable(
    monkeypatch,
) -> None:
    def _fake_resolve(_canonical_model, revision="main"):
        return "resolved-hash" if revision == "main" else None

    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", _fake_resolve)

    assert (
        semantic._resolve_revision_for_cache("some-generic-model", None, strict=True)
        == "resolved-hash"
    )
    # A branch/tag that cannot be mapped offline disables caching in strict mode
    # (never in loose mode, where the same input keys by its label instead).
    assert (
        semantic._resolve_revision_for_cache("some-generic-model", "feature-branch", strict=True)
        is None
    )
    assert (
        semantic._resolve_revision_for_cache("some-generic-model", "feature-branch", strict=False)
        == "feature-branch"
    )


def test_confirm_cache_revision_after_load_loose_opt_in_trusts_pre_load_label() -> None:
    # Loose mode never inspects the model at all: an arbitrary object without
    # the introspection surface _get_loaded_model_commit_hash expects proves
    # that no post-load reconciliation happens.
    sentinel_model = object()

    assert (
        semantic._confirm_cache_revision_after_load(
            sentinel_model, "some-generic-model", "main", strict=False
        )
        == "main"
    )
    assert (
        semantic._confirm_cache_revision_after_load(
            sentinel_model, "some-generic-model", None, strict=False
        )
        == "main"
    )
    commit_hash = "b" * 40
    assert (
        semantic._confirm_cache_revision_after_load(
            sentinel_model, "some-generic-model", commit_hash, strict=False
        )
        == commit_hash
    )


def test_confirm_cache_revision_after_load_strict_requires_loaded_commit(monkeypatch) -> None:
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    assert (
        semantic._confirm_cache_revision_after_load(
            object(), "some-generic-model", "main", strict=True
        )
        is None
    )

    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "loaded-hash")

    assert (
        semantic._confirm_cache_revision_after_load(
            object(), "some-generic-model", "main", strict=True
        )
        == "loaded-hash"
    )


def test_loose_opt_in_cache_survives_simulated_branch_move(tmp_path: Path, monkeypatch) -> None:
    """A warm loose-mode cache must not invalidate when an upstream ref moves.

    Uses an unpinned (generic-profile) model name so the default request
    genuinely goes through the symbolic-revision path rather than a built-in
    profile's pinned commit hash. The label ("main") is the whole key; there
    is no hub-cache lookup to go stale, so re-pointing what a branch would
    resolve to has no effect.
    """
    units = extract_arithmetic_units(tmp_path)
    model = WarmCacheModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_a, **_k: model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "a" * 40)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("loose mode must never consult the offline hub-cache lookup")

    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", fail_if_called)

    first = compute_embeddings(
        units,
        model_name="some-generic-model",
        device="cpu",
        cache_scope=tmp_path,
        strict_revision_cache=False,
    )
    assert model.encode_calls == 1

    second = compute_embeddings(
        units,
        model_name="some-generic-model",
        device="cpu",
        cache_scope=tmp_path,
        strict_revision_cache=False,
    )

    assert model.encode_calls == 1
    np.testing.assert_array_equal(first, second)


def test_default_revision_cache_reencodes_after_simulated_branch_move(
    tmp_path: Path, monkeypatch
) -> None:
    """A moved branch changes the resolved commit hash, invalidating a strict-mode warm cache.

    Uses an unpinned (generic-profile) model name with no explicit revision,
    so the resolved commit is entirely a function of the (stubbed) offline
    hub-cache lookup and the (stubbed) post-load reported commit - both are
    moved together to simulate a real branch move.
    """
    units = extract_arithmetic_units(tmp_path)
    model = WarmCacheModel()
    commit_a = "a" * 40
    commit_b = "b" * 40
    loaded_revisions: list[str | None] = []

    def fake_get_model(*_args, **kwargs):
        loaded_revisions.append(kwargs.get("revision"))
        return model

    monkeypatch.setattr(semantic, "get_model", fake_get_model)
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda *_a, **_k: commit_a)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: commit_a)

    compute_embeddings(
        units,
        model_name="some-generic-model",
        device="cpu",
        cache_scope=tmp_path,
    )
    assert model.encode_calls == 1

    # Same resolved commit: a warm hit needs no model load at all.
    compute_embeddings(
        units,
        model_name="some-generic-model",
        device="cpu",
        cache_scope=tmp_path,
    )
    assert model.encode_calls == 1

    # Simulate an upstream branch move to a new commit.
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda *_a, **_k: commit_b)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: commit_b)

    compute_embeddings(
        units,
        model_name="some-generic-model",
        device="cpu",
        cache_scope=tmp_path,
    )

    assert model.encode_calls == 2
    assert loaded_revisions == [commit_a, commit_b]


def test_default_revision_cache_disables_caching_for_unmappable_symbolic_ref(
    tmp_path: Path, monkeypatch
) -> None:
    units = extract_arithmetic_units(tmp_path)
    model = WarmCacheModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_a, **_k: model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda *_a, **_k: None)

    for _ in range(2):
        compute_embeddings(
            units,
            model_name="gte-modernbert-base",
            revision="unmappable-branch",
            device="cpu",
            cache_scope=tmp_path,
        )

    assert model.encode_calls == 2


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def f():\r\n    return 1\r\n", "def f():\n    return 1"),
        ("def f():\r    return 1\r", "def f():\n    return 1"),
        ("\r\n  keep  \r\n", "keep"),
        ("already\nnormalized", "already\nnormalized"),
    ],
)
def test_prepare_embedding_text_normalizes_line_endings(source: str, expected: str) -> None:
    assert semantic._prepare_embedding_text(source) == expected


def test_crlf_and_lf_units_share_one_embedding_cache_key(tmp_path: Path) -> None:
    lf_source = "fn f() -> i64 {\n    1\n}\n"
    lf_unit = CodeUnit(
        name="f",
        qualified_name="sample::f",
        unit_type=CodeUnitType.FUNCTION,
        file_path=tmp_path / "lf.rs",
        lineno=1,
        end_lineno=3,
        source=lf_source,
        language="rust",
    )
    crlf_unit = CodeUnit(
        name="f",
        qualified_name="sample::f",
        unit_type=CodeUnitType.FUNCTION,
        file_path=tmp_path / "crlf.rs",
        lineno=1,
        end_lineno=3,
        source=lf_source.replace("\n", "\r\n"),
        language="rust",
    )

    lf_text = semantic._prepare_embedding_text(lf_unit.source)
    crlf_text = semantic._prepare_embedding_text(crlf_unit.source)
    assert lf_text == crlf_text

    lf_key = compute_cache_key("model", "revision", lf_text)
    assert lf_key == compute_cache_key("model", "revision", crlf_text)
    # Without normalization the two checkouts would key - and embed - apart.
    assert lf_key != compute_cache_key("model", "revision", crlf_unit.source.strip())


def test_compute_embeddings_normalizes_crlf_before_encoding(tmp_path: Path, monkeypatch) -> None:
    unit = CodeUnit(
        name="f",
        qualified_name="sample::f",
        unit_type=CodeUnitType.FUNCTION,
        file_path=tmp_path / "sample.rs",
        lineno=1,
        end_lineno=3,
        source="fn f() -> i64 {\r\n    1\r\n}\r\n",
        language="rust",
    )
    model = RecordingModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_a, **_k: model)

    compute_embeddings([unit], device="cpu")

    assert model.encoded == [["fn f() -> i64 {\n    1\n}"]]
