"""Revision keying and cache identity: dtype/runtime/fast-math variants, provenance, and local-model fingerprinting."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from codedupes import embedding_cache, semantic
from codedupes.embedding_cache import EmbeddingCache
from codedupes.semantic import (
    compute_embeddings,
    compute_embeddings_with_identity,
)
from tests.conftest import extract_units
from tests.embedding_cache_helpers import (
    REVISION_1,
    REVISION_2,
    CountingModel,
    MidEncodeCpuFallbackModel,
    five_units,
    patch_get_model,
)


def test_full_cache_hit_skips_model_load_and_encode(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model = CountingModel()
    get_model_counts = patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    first = compute_embeddings(
        units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path
    )
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1
    assert len(model.encode_calls[0]) == 5

    second = compute_embeddings(
        units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path
    )
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("embed", [compute_embeddings, compute_embeddings_with_identity])
@pytest.mark.parametrize("cache_state", ["cold", "warm"])
@pytest.mark.parametrize(
    ("unit_count", "document_count"),
    [(2, 1), (1, 2), (0, 1)],
    ids=["shorter", "longer", "empty-corpus"],
)
def test_document_text_count_rejected_before_cache_or_model_work(
    tmp_path, monkeypatch, embed, cache_state, unit_count, document_count
) -> None:
    units = five_units(tmp_path)[:2]
    texts = tuple(f"path: mod.py\n{unit.source}" for unit in units)
    model = CountingModel()
    patch_get_model(monkeypatch, model)
    options = {
        "model_name": "test-model",
        "revision": REVISION_1,
        "device": "cpu",
        "cache_scope": tmp_path,
        "use_cache": True,
        "search_document": "contextual",
    }
    if cache_state == "warm":
        first, _ = compute_embeddings_with_identity(units, document_texts=texts, **options)
        warm, _ = compute_embeddings_with_identity(units, document_texts=texts, **options)
        np.testing.assert_array_equal(first, warm)
        assert first.shape[0] == len(units)
        assert len(model.encode_calls) == 1

    def unexpected_work(*_args, **_kwargs):
        pytest.fail("Mismatched document texts must be rejected before cache/model work")

    monkeypatch.setattr(semantic, "_resolve_revision_for_cache", unexpected_work)
    monkeypatch.setattr(semantic, "_prepare_cache_context", unexpected_work)
    monkeypatch.setattr(semantic, "get_model", unexpected_work)
    with pytest.raises(ValueError, match="document_texts must have the same length as units"):
        embed(units[:unit_count], document_texts=texts[:document_count], **options)


def test_strict_symbolic_revision_revalidates_before_cache_hit(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model = CountingModel()
    get_model_counts = patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda *_args: None)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    first = compute_embeddings(
        units,
        model_name="test-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=True,
    )
    second = compute_embeddings(
        units,
        model_name="test-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=True,
    )

    assert get_model_counts["count"] == 2
    assert len(model.encode_calls) == 2
    np.testing.assert_array_equal(first, second)


# The mps/cpu decision-table rows for _dtype_variant_for are authoritative in
# tests/test_semantic_model_loading.py; this file keeps only the cache-key
# consequence of a resolved non-default dtype (the branch that decision table
# does not reach, since it never mocks a concrete resolved device).
def test_embeddinggemma_cache_variant_scopes_only_nondefault_dtype(monkeypatch):
    profile = semantic.resolve_model_profile("embeddinggemma-300m")
    monkeypatch.setattr(
        semantic,
        "_resolve_semantic_device_request",
        lambda *_args, **_kwargs: "cuda",
    )
    selected_dtype = {"value": "torch.bfloat16"}
    monkeypatch.setattr(
        semantic,
        "_resolve_model_dtype",
        lambda _family, _device, **_kwargs: selected_dtype["value"],
    )

    assert semantic._dtype_variant_for(profile, "cuda", mps_fallback=None) == "dtype=torch.bfloat16"

    selected_dtype["value"] = "torch.float32"
    assert semantic._dtype_variant_for(profile, "cuda", mps_fallback=None) == ""


def test_runtime_upgrade_invalidates_whole_corpus_not_row_subset(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 1

    # Same model, revision, and texts — only the installed inference stack differs.
    monkeypatch.setattr(semantic, "_safe_package_version", lambda _name: "99.0.0-upgraded")

    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 2
    # Every unit re-embedded: no partial reuse of vectors from the old runtime.
    assert len(model.encode_calls[-1]) == len(units)


def test_numpy_upgrade_changes_embedding_runtime_fingerprint(monkeypatch):
    versions = {
        package: semantic._safe_package_version(package) or "missing"
        for package in ("numpy", "torch", "transformers", "tokenizers", "sentence-transformers")
    }
    monkeypatch.setattr(semantic, "_safe_package_version", versions.get)

    before = semantic._embedding_runtime_fingerprint()
    versions["numpy"] = f"{versions['numpy']}+different"

    assert semantic._embedding_runtime_fingerprint() != before


def test_cache_variant_includes_encode_plan_identity():
    profile = semantic.resolve_model_profile("test-model")
    plain = semantic._cache_variant_for(
        profile, "cpu", semantic.EncodePlan(route="symmetric"), mps_fallback=None
    )
    prompted = semantic._cache_variant_for(
        profile, "cpu", semantic.EncodePlan(route="symmetric", prompt="custom: "), mps_fallback=None
    )
    routed = semantic._cache_variant_for(
        profile, "cpu", semantic.EncodePlan(route="document"), mps_fallback=None
    )
    assert len({plain, prompted, routed}) == 3


def test_cache_variant_keys_mps_fast_math_policy(monkeypatch):
    profile = semantic.resolve_model_profile("test-model")
    plan = semantic.EncodePlan(route="symmetric")

    monkeypatch.delenv("PYTORCH_MPS_FAST_MATH", raising=False)
    baseline = semantic._cache_variant_for(profile, "mps", plan, mps_fallback=None)
    cpu_baseline = semantic._cache_variant_for(profile, "cpu", plan, mps_fallback=None)

    # Disabled fast math is the same policy as an unset variable.
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "0")
    assert semantic._cache_variant_for(profile, "mps", plan, mps_fallback=None) == baseline

    # An enabled policy must split the key space wherever MPS can execute.
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    assert semantic._cache_variant_for(profile, "mps", plan, mps_fallback=None) != baseline

    # torch enables fast math for any set value except the literal "0": the
    # empty string and whitespace-wrapped zeros must key as fast-math variants.
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "")
    assert semantic._cache_variant_for(profile, "mps", plan, mps_fallback=None) != baseline
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", " 0")
    assert semantic._cache_variant_for(profile, "mps", plan, mps_fallback=None) != baseline
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")

    # Devices that can never execute Metal kernels ignore the policy.
    assert semantic._cache_variant_for(profile, "cpu", plan, mps_fallback=None) == cpu_baseline

    # On macOS, ``auto`` can resolve to MPS, so it splits with the policy too.
    monkeypatch.setattr(sys, "platform", "darwin")
    auto_fast = semantic._cache_variant_for(profile, "auto", plan, mps_fallback=None)
    monkeypatch.delenv("PYTORCH_MPS_FAST_MATH")
    assert semantic._cache_variant_for(profile, "auto", plan, mps_fallback=None) != auto_fast


def test_mps_fast_math_policy_change_invalidates_warm_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("PYTORCH_MPS_FAST_MATH", raising=False)
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 1

    # Same request under an altered Metal math policy: faithful-float32 rows
    # must not satisfy hits, so the whole corpus re-embeds. The fake reports
    # MPS execution so the fast-math key space accepts its writes.
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    model.device = "mps"
    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 2
    assert len(model.encode_calls[-1]) == len(units)

    # The fast-math key space warms independently.
    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 2


def test_fast_math_variant_restarts_under_faithful_cpu_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    # The request is keyed for fast math, but execution lands on CPU. Corpus
    # assembly restarts under the faithful policy and publishes that complete
    # matrix to the CPU/MPS-float32 key space.
    model.device = "cpu"
    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 1

    # Another fallback-shaped request reuses the complete faithful matrix.
    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 1


def test_fast_math_variant_rebuilds_corpus_after_mid_encode_cpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    units = five_units(tmp_path)
    model = MidEncodeCpuFallbackModel()
    model.device = "mps"
    patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    # Execution starts on MPS but lands on CPU mid-encode. The first vectors are
    # discarded and every row is encoded again under the faithful CPU policy.
    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 2
    assert all(len(call) == len(units) for call in model.encode_calls)

    # A later request cannot consume the faithful rows as fast-math hits. This
    # model lands on CPU again, after which the faithful restart is already warm.
    model.device = "mps"
    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 3


def test_compute_embeddings_passes_raw_text_with_prompt_config(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)

    compute_embeddings(
        units,
        model_name="test-model",
        instruction_prefix="custom: ",
        cache_scope=None,
    )

    # The instruction travels as the backend prompt; input texts stay raw.
    assert model.encode_calls == [[unit.source.strip() for unit in units]]
    assert model.prompts_seen == ["custom: "]


def test_cache_key_sensitive_to_model_revision_prefix_and_task(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    base: dict[str, object] = {
        "model_name": "embeddinggemma-300m",
        "revision": REVISION_1,
        "cache_scope": tmp_path,
    }
    compute_embeddings(units, **base)
    assert len(model.encode_calls) == 1

    compute_embeddings(units, **{**base, "revision": REVISION_2})
    assert len(model.encode_calls) == 2

    compute_embeddings(units, **{**base, "model_name": "other-model"})
    assert len(model.encode_calls) == 3

    compute_embeddings(units, **{**base, "instruction_prefix": "CUSTOM: "})
    assert len(model.encode_calls) == 4

    compute_embeddings(units, **{**base, "semantic_task": "classification"})
    assert len(model.encode_calls) == 5


def test_strict_revision_drift_after_model_load_discards_stale_prefetched_hits(
    tmp_path, monkeypatch
):
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)

    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda _model: "rev-a")
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "rev-a")
    compute_embeddings(
        units,
        model_name="drift-model",
        revision=None,
        cache_scope=tmp_path,
        strict_revision_cache=True,
    )
    assert len(model.encode_calls) == 1
    assert len(model.encode_calls[0]) == 5

    changed = copy.copy(units[1])
    changed.source = "def other(x):\n    return x + 777\n"
    mixed_units = [units[0], changed, units[2]]
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "rev-b")

    result = compute_embeddings(
        mixed_units,
        model_name="drift-model",
        revision=None,
        cache_scope=tmp_path,
        strict_revision_cache=True,
    )

    assert len(model.encode_calls) == 2
    assert len(model.encode_calls[-1]) == 3
    assert result.shape == (3, model.dim)


def test_confirm_source_commit_matches_and_purges_on_drift(tmp_path):
    cache = EmbeddingCache(tmp_path)
    scope = tmp_path / "repo"
    scope.mkdir()
    key = embedding_cache.compute_cache_key("some/model", "main", "text-a")
    cache.put_many(
        scope,
        "some/model",
        "main",
        [(key, np.array([1.0, 0.0], dtype=np.float32))],
        expected_source_commit="a" * 40,
    )

    # The write stamped provenance; a matching commit confirms and keeps hits.
    assert cache.confirm_source_commit(scope, "some/model", "main", "a" * 40) is True
    assert set(cache.get_many(scope, "some/model", "main", [key])) == {key}

    # A different loaded commit is drift: the whole label shard is purged.
    assert cache.confirm_source_commit(scope, "some/model", "main", "b" * 40) is False
    assert cache.get_many(scope, "some/model", "main", [key]) == {}
    assert cache.confirm_source_commit(scope, "some/model", "main", "b" * 40) is True


@pytest.mark.parametrize("existing_commit", [None, "a" * 40])
def test_put_many_rejects_unverified_provenance_required_batch(tmp_path, existing_commit):
    cache = EmbeddingCache(tmp_path)
    scope = tmp_path / "repo"
    scope.mkdir()
    vector = np.array([1.0, 0.0], dtype=np.float32)
    if existing_commit is not None:
        cache.put_many(
            scope,
            "some/model",
            "main",
            [("corpus", vector)],
            expected_source_commit=existing_commit,
        )
    shard_dir = cache.shard_dir(scope, "some/model", "main")
    before = embedding_cache._read_shard_meta(shard_dir)

    cache.put_many(
        scope,
        "some/model",
        "main",
        [("query", vector)],
        namespace="query",
        require_source_commit=True,
    )

    assert cache.get_many(scope, "some/model", "main", ["query"]) == {}
    assert embedding_cache._read_shard_meta(shard_dir) == before


def test_get_many_with_provenance_returns_the_snapshot_commit(tmp_path):
    cache = EmbeddingCache(tmp_path)
    scope = tmp_path / "repo"
    scope.mkdir()
    key = embedding_cache.compute_cache_key("some/model", "main", "text-a")
    vector = np.array([1.0, 0.0], dtype=np.float32)

    empty = cache.get_many_with_provenance(scope, "some/model", "main", [key])
    assert empty.vectors == {}
    assert empty.source_commit is None

    cache.put_many(scope, "some/model", "main", [(key, vector)], expected_source_commit="a" * 40)
    lookup = cache.get_many_with_provenance(scope, "some/model", "main", [key])
    assert set(lookup.vectors) == {key}
    assert lookup.source_commit == "a" * 40

    # Immutable-revision shards report no provenance: the revision is the truth.
    cache.put_many(scope, "some/model", REVISION_1, [(key, vector)])
    pinned = cache.get_many_with_provenance(scope, "some/model", REVISION_1, [key])
    assert set(pinned.vectors) == {key}
    assert pinned.source_commit is None


def test_loose_corpus_writes_stamp_the_loaded_commit(tmp_path, monkeypatch):
    """The semantic write path must hand its loaded commit to the cache, so the
    TOCTOU rejection in put_many has a truth to compare against."""
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "a" * 40)

    compute_embeddings(
        units,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=False,
    )

    cache = embedding_cache.get_embedding_cache()
    assert cache is not None
    canonical = semantic.resolve_model_profile("drift-model").canonical_name
    meta = embedding_cache._read_shard_meta(cache.shard_dir(tmp_path, canonical, "main"))
    assert meta is not None
    assert meta["source_commit"] == "a" * 40


def test_pinned_revision_shards_skip_the_source_commit_guard(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)
    # Immutable commit-hash keys can never drift, whatever the backend reports.
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: REVISION_2)

    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)

    changed = copy.copy(units[1])
    changed.source = "def other(x):\n    return x + 777\n"
    result = compute_embeddings(
        [units[0], changed, units[2]],
        model_name="test-model",
        revision=REVISION_1,
        cache_scope=tmp_path,
    )

    assert len(model.encode_calls[-1]) == 1
    assert result.shape == (3, model.dim)


def test_revision_is_mutable_label_classification(tmp_path, monkeypatch):
    assert semantic._revision_is_mutable_label("test-model", "main") is True
    assert semantic._revision_is_mutable_label("test-model", REVISION_1) is False
    assert semantic._revision_is_mutable_label("test-model", None) is False

    monkeypatch.setattr(semantic, "resolve_local_model_path", lambda _name: tmp_path)
    assert semantic._revision_is_mutable_label("test-model", "content-fingerprint") is False


def test_strict_unconfirmable_loaded_revision_disables_cache(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model = CountingModel()
    get_model_counts = patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda _model: "rev-a")
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    first = compute_embeddings(
        units, model_name="test-model", cache_scope=tmp_path, strict_revision_cache=True
    )
    assert len(model.encode_calls) == 1

    second = compute_embeddings(
        units, model_name="test-model", cache_scope=tmp_path, strict_revision_cache=True
    )
    assert get_model_counts["count"] == 2
    assert len(model.encode_calls) == 2
    assert EmbeddingCache().stats()["entries"] == 0
    np.testing.assert_array_equal(first, second)


def _fake_local_model_dir(tmp_path: Path, name: str = "gemma-work-copy") -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    (model_dir / "model.safetensors").write_text("weights-v1")
    return model_dir


@pytest.mark.parametrize(
    "asset_name",
    ["model.safetensors", "license_head.safetensors", "notice_tokens.json", "readme_encoder.py"],
)
def test_local_model_dir_cache_uses_fingerprint_not_revision(tmp_path, monkeypatch, asset_name):
    units = five_units(tmp_path)
    model = CountingModel()
    get_model_counts = patch_get_model(monkeypatch, model)
    model_dir = _fake_local_model_dir(tmp_path)
    asset = model_dir / asset_name
    asset.write_text("weights-v1", encoding="utf-8")

    compute_embeddings(
        units,
        model_name=str(model_dir),
        revision="requested-revision-a",
        cache_scope=tmp_path,
    )
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1

    second = compute_embeddings(
        units,
        model_name=str(model_dir),
        revision="requested-revision-b",
        cache_scope=tmp_path,
    )
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1
    assert second.shape == (5, 4)

    # Replacing an embedding asset in place must change the fingerprint revision and
    # invalidate every cached vector for this model directory.
    asset.write_text("weights-v2-longer", encoding="utf-8")
    compute_embeddings(
        units,
        model_name=str(model_dir),
        revision="requested-revision-c",
        cache_scope=tmp_path,
    )
    assert get_model_counts["count"] == 2
    assert len(model.encode_calls) == 2
    assert len(model.encode_calls[-1]) == 5


def test_local_model_dir_relative_and_absolute_share_cache(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model = CountingModel()
    get_model_counts = patch_get_model(monkeypatch, model)
    model_dir = _fake_local_model_dir(tmp_path)

    compute_embeddings(units, model_name=str(model_dir), cache_scope=tmp_path)
    assert get_model_counts["count"] == 1

    monkeypatch.chdir(tmp_path)
    compute_embeddings(units, model_name="./gemma-work-copy", cache_scope=tmp_path)
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1


def test_fingerprint_local_model_dir_stability_and_edge_cases(tmp_path):
    model_dir = _fake_local_model_dir(tmp_path)
    first = semantic._fingerprint_local_model_dir(model_dir)
    assert first is not None and first.startswith("dir-")
    assert semantic._fingerprint_local_model_dir(model_dir) == first

    hf_metadata = model_dir / ".cache" / "huggingface" / "download"
    hf_metadata.mkdir(parents=True)
    (hf_metadata / "config.json.metadata").write_text("updated-download-metadata")
    assert semantic._fingerprint_local_model_dir(model_dir) == first

    empty = tmp_path / "empty-model"
    empty.mkdir()
    assert semantic._fingerprint_local_model_dir(empty) is None


def test_local_model_content_change_with_preserved_size_and_mtime_invalidates(
    tmp_path, monkeypatch
):
    # A byte-for-byte-length rewrite with a restored mtime defeats a
    # metadata-only fingerprint; the content-backed fingerprint must still miss.
    units = five_units(tmp_path)
    model = CountingModel()
    get_model_counts = patch_get_model(monkeypatch, model)
    model_dir = _fake_local_model_dir(tmp_path)
    weights_path = model_dir / "model.safetensors"
    original_stat = weights_path.stat()

    compute_embeddings(units, model_name=str(model_dir), cache_scope=tmp_path)
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1

    weights_path.write_text("weights-v2")
    os.utime(weights_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    rewritten_stat = weights_path.stat()
    assert rewritten_stat.st_size == original_stat.st_size
    assert rewritten_stat.st_mtime_ns == original_stat.st_mtime_ns

    compute_embeddings(units, model_name=str(model_dir), cache_scope=tmp_path)
    assert get_model_counts["count"] == 2
    assert len(model.encode_calls) == 2
    assert len(model.encode_calls[-1]) == 5


def test_local_model_swap_during_load_discards_preload_hits(tmp_path, monkeypatch):
    # Weights swapped between key derivation and model load must not let
    # vectors cached for the old weights survive into the new model's matrix.
    units = five_units(tmp_path)
    model = CountingModel()
    model_dir = _fake_local_model_dir(tmp_path)
    weights_path = model_dir / "model.safetensors"
    get_model_counts = {"count": 0}

    def swapping_get_model(*_args, **_kwargs):
        get_model_counts["count"] += 1
        if get_model_counts["count"] == 2:
            weights_path.write_text("weights-v2-swapped-mid-load")
        return model

    monkeypatch.setattr(semantic, "get_model", swapping_get_model)

    compute_embeddings(units, model_name=str(model_dir), cache_scope=tmp_path)
    assert len(model.encode_calls) == 1

    extra_source = "def brand_new():\n    return 99\n"
    all_units = units + extract_units(tmp_path, extra_source, filename="extra.py")
    second = compute_embeddings(all_units, model_name=str(model_dir), cache_scope=tmp_path)
    assert second.shape[0] == 6
    assert get_model_counts["count"] == 2
    # All six units re-embed with the freshly loaded weights; zero stale reuse.
    assert len(model.encode_calls[-1]) == 6

    third = compute_embeddings(all_units, model_name=str(model_dir), cache_scope=tmp_path)
    assert third.shape[0] == 6
    assert get_model_counts["count"] == 2
    assert len(model.encode_calls) == 2


def test_fingerprint_manifest_reuses_digests_and_ignores_touch(tmp_path, monkeypatch):
    model_dir = _fake_local_model_dir(tmp_path)
    hash_calls = {"count": 0}
    real_hash = semantic._hash_file_content

    def counting_hash(path):
        hash_calls["count"] += 1
        return real_hash(path)

    monkeypatch.setattr(semantic, "_hash_file_content", counting_hash)

    first = semantic._fingerprint_local_model_dir(model_dir)
    assert first is not None
    assert hash_calls["count"] == 2

    assert semantic._fingerprint_local_model_dir(model_dir) == first
    assert hash_calls["count"] == 2

    # A metadata-only touch rehashes that file, but the fingerprint is
    # content-based so cached vectors survive.
    os.utime(model_dir / "model.safetensors")
    assert semantic._fingerprint_local_model_dir(model_dir) == first
    assert hash_calls["count"] == 3

    # The on-disk manifest keeps a fresh process cheap: drop the in-process
    # memo and confirm nothing is rehashed.
    with semantic._local_model_manifest_lock:
        semantic._local_model_manifest_memo.pop(str(model_dir), None)
    assert semantic._fingerprint_local_model_dir(model_dir) == first
    assert hash_calls["count"] == 3

    (model_dir / "model.safetensors").write_text("weights-v2-longer")
    assert semantic._fingerprint_local_model_dir(model_dir) != first


def test_model_slug_for_local_paths_is_bounded_and_collision_safe():
    hub = embedding_cache._model_slug("Alibaba-NLP/gte-modernbert-base")
    assert hub == "Alibaba-NLP--gte-modernbert-base"

    deep = "/very/deep/nested/path/to/models/gte-modernbert-base"
    other = "/other/location/gte-modernbert-base"
    slug_a = embedding_cache._model_slug(deep)
    slug_b = embedding_cache._model_slug(other)
    assert slug_a.startswith("local--gte-modernbert-base-")
    assert "/" not in slug_a
    assert slug_a != slug_b
    assert len(slug_a) < 60
