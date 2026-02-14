from __future__ import annotations

from codedupes.semantic_profiles import (
    DEFAULT_C2LLM_REVISION,
    get_default_semantic_threshold,
    list_supported_models,
    resolve_model_name,
    resolve_model_profile,
)


def test_resolve_builtin_model_aliases_to_canonical_ids() -> None:
    assert resolve_model_name("gte-modernbert-base") == "Alibaba-NLP/gte-modernbert-base"
    assert resolve_model_name("c2llm-0.5b") == "codefuse-ai/C2LLM-0.5B"
    assert resolve_model_name("embeddinggemma-300m") == "unsloth/embeddinggemma-300m"
    assert resolve_model_name("google/embeddinggemma-300m") == "unsloth/embeddinggemma-300m"


def test_unknown_model_uses_generic_fallback_profile() -> None:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    profile = resolve_model_profile(model_name)
    assert profile.family == "generic"
    assert profile.canonical_name == model_name
    assert profile.default_trust_remote_code is False


def test_c2llm_profile_defaults() -> None:
    profile = resolve_model_profile("c2llm-0.5b")
    assert profile.family == "c2llm"
    assert profile.default_revision == DEFAULT_C2LLM_REVISION
    assert profile.default_trust_remote_code is True
    assert profile.requires_deepspeed is True


def test_dynamic_c2llm_profile_for_non_builtin_model() -> None:
    profile = resolve_model_profile("codefuse-ai/C2LLM-7B")
    assert profile.family == "c2llm"
    assert profile.canonical_name == "codefuse-ai/C2LLM-7B"
    assert profile.default_revision is None
    assert profile.default_trust_remote_code is True


def test_supported_model_list_contains_three_profiles() -> None:
    keys = [profile.key for profile in list_supported_models()]
    assert keys == ["gte-modernbert-base", "c2llm-0.5b", "embeddinggemma-300m"]


def test_model_threshold_lookup_works_for_builtin_and_unknown() -> None:
    assert get_default_semantic_threshold("gte-modernbert-base") > 0
    assert get_default_semantic_threshold("unknown/model-id") > 0
