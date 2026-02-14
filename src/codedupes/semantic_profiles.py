"""Model profile registry for semantic embedding backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SemanticModelFamily = Literal["gte-modernbert", "c2llm", "embeddinggemma", "generic"]

DEFAULT_FALLBACK_SEMANTIC_THRESHOLD = 0.82
DEFAULT_C2LLM_REVISION = "bd6d0ddb29f0c9a3d0f14281aedc9f940bb8d67a"


@dataclass(frozen=True)
class SemanticModelProfile:
    """Semantic embedding model profile."""

    key: str
    canonical_name: str
    aliases: tuple[str, ...]
    family: SemanticModelFamily
    default_revision: str | None = None
    default_trust_remote_code: bool = False
    requires_deepspeed: bool = False
    left_padding: bool = False
    low_cpu_mem_usage: bool = False
    default_semantic_threshold: float = DEFAULT_FALLBACK_SEMANTIC_THRESHOLD

    def all_aliases(self) -> tuple[str, ...]:
        """Return all user-facing names that map to this profile."""
        return (self.key, self.canonical_name, *self.aliases)


_BUILTIN_MODEL_PROFILES: tuple[SemanticModelProfile, ...] = (
    SemanticModelProfile(
        key="gte-modernbert-base",
        canonical_name="Alibaba-NLP/gte-modernbert-base",
        aliases=(
            "gte-modernbert",
            "alibaba-nlp/gte-modernbert-base",
        ),
        family="gte-modernbert",
        default_semantic_threshold=0.96,
    ),
    SemanticModelProfile(
        key="c2llm-0.5b",
        canonical_name="codefuse-ai/C2LLM-0.5B",
        aliases=(
            "codefuse-ai/c2llm-0.5b",
            "codefuse-ai/C2LLM-0.5B",
        ),
        family="c2llm",
        default_revision=DEFAULT_C2LLM_REVISION,
        default_trust_remote_code=True,
        requires_deepspeed=True,
        left_padding=True,
        low_cpu_mem_usage=True,
        default_semantic_threshold=0.80,
    ),
    SemanticModelProfile(
        key="embeddinggemma-300m",
        canonical_name="unsloth/embeddinggemma-300m",
        aliases=(
            "unsloth/embeddinggemma-300m",
            "google/embeddinggemma-300m",
            "embeddinggemma",
        ),
        family="embeddinggemma",
        default_semantic_threshold=0.86,
    ),
)

_GENERIC_PROFILE = SemanticModelProfile(
    key="generic",
    canonical_name="",
    aliases=(),
    family="generic",
)


def _normalize_model_key(value: str) -> str:
    """Normalize model aliases for stable lookup."""
    return value.strip().lower()


def list_supported_models() -> list[SemanticModelProfile]:
    """Return the built-in model profiles in deterministic order."""
    return list(_BUILTIN_MODEL_PROFILES)


def _builtin_alias_map() -> dict[str, SemanticModelProfile]:
    """Return normalized alias map for built-in profiles."""
    alias_map: dict[str, SemanticModelProfile] = {}
    for profile in _BUILTIN_MODEL_PROFILES:
        for alias in profile.all_aliases():
            alias_map[_normalize_model_key(alias)] = profile
    return alias_map


def _build_dynamic_c2llm_profile(model_name: str) -> SemanticModelProfile:
    """Build a C2LLM-family profile for non-builtin C2LLM model IDs."""
    return SemanticModelProfile(
        key=model_name,
        canonical_name=model_name,
        aliases=(),
        family="c2llm",
        default_revision=None,
        default_trust_remote_code=True,
        requires_deepspeed=True,
        left_padding=True,
        low_cpu_mem_usage=True,
    )


def resolve_model_profile(model_name: str) -> SemanticModelProfile:
    """Resolve a user model identifier into a concrete model profile."""
    alias_map = _builtin_alias_map()
    normalized = _normalize_model_key(model_name)
    builtin = alias_map.get(normalized)
    if builtin is not None:
        return builtin

    if "c2llm" in normalized:
        return _build_dynamic_c2llm_profile(model_name)

    return SemanticModelProfile(
        key=model_name,
        canonical_name=model_name,
        aliases=(),
        family=_GENERIC_PROFILE.family,
        default_semantic_threshold=_GENERIC_PROFILE.default_semantic_threshold,
    )


def resolve_model_name(model_name: str) -> str:
    """Resolve model name aliases to canonical model IDs."""
    return resolve_model_profile(model_name).canonical_name


def get_default_semantic_threshold(model_name: str) -> float:
    """Return semantic threshold default for the resolved model profile."""
    return resolve_model_profile(model_name).default_semantic_threshold
