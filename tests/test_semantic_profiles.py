from __future__ import annotations

from pathlib import Path

from codedupes.semantic_profiles import (
    DEFAULT_FALLBACK_SEARCH_THRESHOLD,
    get_default_search_threshold,
    get_default_semantic_threshold,
    list_supported_models,
    resolve_local_model_path,
    resolve_model_profile,
)


def test_resolve_builtin_model_aliases_to_canonical_ids() -> None:
    expected = {
        "gte-modernbert-base": "Alibaba-NLP/gte-modernbert-base",
        "embeddinggemma-300m": "unsloth/embeddinggemma-300m",
        "google/embeddinggemma-300m": "unsloth/embeddinggemma-300m",
    }
    assert {alias: resolve_model_profile(alias).canonical_name for alias in expected} == expected


def test_unknown_model_uses_generic_fallback_profile() -> None:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    profile = resolve_model_profile(model_name)
    assert profile.family == "generic"
    assert profile.canonical_name == model_name
    assert profile.default_trust_remote_code is False


def test_supported_model_list_contains_two_profiles() -> None:
    keys = [profile.key for profile in list_supported_models()]
    assert keys == ["gte-modernbert-base", "embeddinggemma-300m"]


def test_model_threshold_lookup_works_for_builtin_and_unknown() -> None:
    assert get_default_semantic_threshold("gte-modernbert-base") > 0
    assert get_default_semantic_threshold("unknown/model-id") > 0


def test_search_threshold_is_looser_than_duplicate_threshold() -> None:
    for profile in list_supported_models():
        assert 0 < profile.default_search_threshold < profile.default_semantic_threshold
    assert get_default_search_threshold("gte-modernbert-base") == 0.50
    assert get_default_search_threshold("unknown/model-id") == DEFAULT_FALLBACK_SEARCH_THRESHOLD


def test_resolve_local_model_path_only_matches_existing_directories(tmp_path: Path) -> None:
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    resolved = resolve_local_model_path(str(model_dir))
    assert resolved == model_dir.resolve()
    assert resolve_local_model_path("Alibaba-NLP/gte-modernbert-base") is None
    assert resolve_local_model_path(str(tmp_path / "missing")) is None
    assert resolve_local_model_path("") is None


def test_local_directory_canonicalizes_to_resolved_path(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "some-finetune"
    model_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    relative = resolve_model_profile("./some-finetune")
    absolute = resolve_model_profile(str(model_dir))
    assert relative.canonical_name == absolute.canonical_name == str(model_dir.resolve())
    assert relative.family == "generic"


def test_local_directory_family_inferred_from_basename(tmp_path: Path) -> None:
    gemma_dir = tmp_path / "embeddinggemma-work-copy"
    gemma_dir.mkdir()
    gemma = resolve_model_profile(str(gemma_dir))
    assert gemma.family == "embeddinggemma"
    assert gemma.canonical_name == str(gemma_dir.resolve())
    builtin = resolve_model_profile("embeddinggemma")
    assert gemma.default_semantic_threshold == builtin.default_semantic_threshold
    assert gemma.default_search_threshold == builtin.default_search_threshold


def test_dynamic_embeddinggemma_profile_for_non_builtin_hub_id() -> None:
    profile = resolve_model_profile("someone/embeddinggemma-300m-code-ft")
    assert profile.family == "embeddinggemma"
    assert profile.canonical_name == "someone/embeddinggemma-300m-code-ft"


def test_dynamic_gte_modernbert_profile_matches_builtin_thresholds(tmp_path: Path) -> None:
    local_dir = tmp_path / "gte-modernbert-base"
    local_dir.mkdir()
    profile = resolve_model_profile(str(local_dir))
    builtin = resolve_model_profile("gte-modernbert-base")
    assert profile.family == "gte-modernbert"
    assert profile.default_semantic_threshold == builtin.default_semantic_threshold
    assert profile.default_search_threshold == builtin.default_search_threshold
