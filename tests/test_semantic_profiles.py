from __future__ import annotations

from pathlib import Path

import pytest

from codedupes.semantic_profiles import (
    DEFAULT_FALLBACK_SEARCH_THRESHOLD,
    DEFAULT_FALLBACK_SEMANTIC_THRESHOLD,
    _true_case_path,
    get_default_search_threshold,
    get_default_semantic_threshold,
    is_explicit_local_model_path,
    list_supported_models,
    resolve_local_model_path,
    resolve_model_profile,
)
from scripts.sweep_hybrid_gates import _resolve_hybrid_semantic_threshold


def _filesystem_is_case_insensitive(tmp_path: Path) -> bool:
    """Detect whether ``tmp_path`` lives on a case-insensitive filesystem.

    :param tmp_path: Writable directory to probe.
    :return: ``True`` if a differently-cased path resolves to the same entry.
    """
    (tmp_path / "CaseProbe").mkdir()
    return (tmp_path / "caseprobe").exists()


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


def test_builtin_profile_alias_lists_are_unique() -> None:
    for profile in list_supported_models():
        aliases = profile.all_aliases()
        assert len(aliases) == len(set(aliases))


def test_model_threshold_lookup_works_for_builtin_and_unknown() -> None:
    assert get_default_semantic_threshold("gte-modernbert-base") > 0
    assert get_default_semantic_threshold("unknown/model-id") > 0


def test_search_threshold_is_looser_than_duplicate_threshold() -> None:
    for profile in list_supported_models():
        assert 0 < profile.default_search_threshold < profile.default_semantic_threshold
    assert get_default_search_threshold("gte-modernbert-base") == 0.50
    assert get_default_search_threshold("embeddinggemma-300m") == 0.40
    assert get_default_search_threshold("unknown/model-id") == DEFAULT_FALLBACK_SEARCH_THRESHOLD


def test_hybrid_sweep_uses_selected_profile_threshold_unless_overridden() -> None:
    assert _resolve_hybrid_semantic_threshold("gte-modernbert-base", None) == 0.96
    assert _resolve_hybrid_semantic_threshold("embeddinggemma-300m", None) == 0.86
    assert _resolve_hybrid_semantic_threshold("gte-modernbert-base", 0.73) == 0.73


def test_resolve_local_model_path_only_matches_existing_directories(tmp_path: Path) -> None:
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    resolved = resolve_local_model_path(str(model_dir))
    assert resolved == model_dir.resolve()
    assert resolve_local_model_path("my-model") is None
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


def test_local_directory_canonicalizes_case_variants_to_one_identity(tmp_path: Path) -> None:
    case_insensitive = _filesystem_is_case_insensitive(tmp_path)
    model_dir = tmp_path / "MyFineTune"
    model_dir.mkdir()

    if not case_insensitive:
        pytest.skip("filesystem is case-sensitive; case variants are distinct paths")

    lower = resolve_model_profile(str(tmp_path / "myfinetune"))
    exact = resolve_model_profile(str(model_dir))
    upper = resolve_model_profile(str(tmp_path / "MYFINETUNE"))

    expected = str(model_dir.resolve())
    assert lower.canonical_name == exact.canonical_name == upper.canonical_name == expected
    assert expected.endswith("MyFineTune")


def test_true_case_path_is_identity_for_exact_case_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "ExactCase"
    model_dir.mkdir()
    assert _true_case_path(model_dir.resolve()) == model_dir.resolve()


def test_true_case_path_corrects_wrong_case_component(tmp_path: Path) -> None:
    if not _filesystem_is_case_insensitive(tmp_path):
        pytest.skip("filesystem is case-sensitive; wrong-case components do not exist on disk")

    model_dir = tmp_path / "TrueCased"
    model_dir.mkdir()
    wrong_case = tmp_path / "truecased"
    assert _true_case_path(wrong_case.resolve()) == model_dir.resolve()


def test_true_case_path_falls_back_gracefully_for_missing_tail(tmp_path: Path) -> None:
    parent = tmp_path / "parent-dir"
    parent.mkdir()
    missing = parent / "does-not-exist"
    assert _true_case_path(missing) == missing


def test_builtin_alias_requires_explicit_path_to_use_same_named_local_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "gte-modernbert-base"
    model_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    builtin = resolve_model_profile("gte-modernbert-base")
    local = resolve_model_profile("./gte-modernbert-base")

    assert builtin.canonical_name == "Alibaba-NLP/gte-modernbert-base"
    assert builtin.default_revision is not None
    assert local.canonical_name == str(model_dir.resolve())
    assert local.family == "gte-modernbert"
    assert local.default_revision is None


def test_builtin_profiles_pin_immutable_revisions() -> None:
    for profile in list_supported_models():
        revision = profile.default_revision
        assert revision is not None
        assert len(revision) == 40
        assert all(character in "0123456789abcdef" for character in revision)


def test_local_directory_family_inferred_from_basename(tmp_path: Path) -> None:
    gemma_dir = tmp_path / "embeddinggemma-work-copy"
    gemma_dir.mkdir()
    gemma = resolve_model_profile(str(gemma_dir))
    assert gemma.family == "embeddinggemma"
    assert gemma.canonical_name == str(gemma_dir.resolve())
    # Family selects loading/prompt behavior only; calibrated thresholds belong
    # to the pinned builtin checkpoint, and an arbitrary local copy may not be it.
    assert gemma.default_semantic_threshold == DEFAULT_FALLBACK_SEMANTIC_THRESHOLD
    assert gemma.default_search_threshold == DEFAULT_FALLBACK_SEARCH_THRESHOLD


def test_hash_named_hf_snapshot_infers_family_from_cache_ancestor(tmp_path: Path) -> None:
    snapshot = (
        tmp_path
        / "models--Alibaba-NLP--gte-modernbert-base"
        / "snapshots"
        / "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
    )
    snapshot.mkdir(parents=True)

    profile = resolve_model_profile(str(snapshot))

    assert profile.family == "gte-modernbert"
    assert profile.default_search_threshold == DEFAULT_FALLBACK_SEARCH_THRESHOLD


def test_arbitrary_local_directory_infers_embeddinggemma_from_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "downloaded-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type": "gemma3_text", "use_bidirectional_attention": true}'
    )

    profile = resolve_model_profile(str(model_dir))

    assert profile.family == "embeddinggemma"
    assert profile.default_semantic_threshold == DEFAULT_FALLBACK_SEMANTIC_THRESHOLD


def test_arbitrary_local_directory_infers_gte_family_from_model_card(tmp_path: Path) -> None:
    model_dir = tmp_path / "downloaded-model"
    model_dir.mkdir()
    (model_dir / "README.md").write_text("# gte-modernbert-base\n\nLocal model copy.\n")

    assert resolve_model_profile(str(model_dir)).family == "gte-modernbert"


def test_explicit_local_path_detection() -> None:
    assert is_explicit_local_model_path("/models/local-copy")
    assert is_explicit_local_model_path("./models/local-copy")
    assert is_explicit_local_model_path("../models/local-copy")
    assert is_explicit_local_model_path("~/models/local-copy")
    assert not is_explicit_local_model_path("models/local-copy")
    assert not is_explicit_local_model_path("local-copy")
    assert not is_explicit_local_model_path("Alibaba-NLP/gte-modernbert-base")


def test_dynamic_embeddinggemma_profile_for_non_builtin_hub_id() -> None:
    profile = resolve_model_profile("someone/embeddinggemma-300m-code-ft")
    assert profile.family == "embeddinggemma"
    assert profile.canonical_name == "someone/embeddinggemma-300m-code-ft"


def test_dynamic_gte_modernbert_profile_keeps_family_but_not_calibration(tmp_path: Path) -> None:
    local_dir = tmp_path / "gte-modernbert-base"
    local_dir.mkdir()
    profile = resolve_model_profile(str(local_dir))
    builtin = resolve_model_profile("gte-modernbert-base")
    assert profile.family == "gte-modernbert"
    assert profile.default_revision is None
    assert profile.default_semantic_threshold != builtin.default_semantic_threshold
    assert profile.default_semantic_threshold == DEFAULT_FALLBACK_SEMANTIC_THRESHOLD
    assert profile.default_search_threshold == DEFAULT_FALLBACK_SEARCH_THRESHOLD
