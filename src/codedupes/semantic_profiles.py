"""Model profile registry for semantic embedding backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SemanticModelFamily = Literal["gte-modernbert", "embeddinggemma", "generic"]

DEFAULT_FALLBACK_SEMANTIC_THRESHOLD = 0.82
DEFAULT_FALLBACK_SEARCH_THRESHOLD = 0.35


@dataclass(frozen=True)
class SemanticModelProfile:
    """Semantic embedding model profile."""

    key: str
    canonical_name: str
    aliases: tuple[str, ...]
    family: SemanticModelFamily
    default_revision: str | None = None
    default_trust_remote_code: bool = False
    default_semantic_threshold: float = DEFAULT_FALLBACK_SEMANTIC_THRESHOLD
    default_search_threshold: float = DEFAULT_FALLBACK_SEARCH_THRESHOLD

    def all_aliases(self) -> tuple[str, ...]:
        """Return all user-facing names that map to this profile.

        :return: Tuple of alias strings including canonical profile keys.
        """
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
        default_search_threshold=0.50,
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
        default_search_threshold=0.40,
    ),
)

_GENERIC_PROFILE = SemanticModelProfile(
    key="generic",
    canonical_name="",
    aliases=(),
    family="generic",
)


def _normalize_model_key(value: str) -> str:
    """Normalize model aliases for stable lookup.

    :param value: Raw model alias.
    :return: Normalized lowercase alias.
    """
    return value.strip().lower()


def list_supported_models() -> list[SemanticModelProfile]:
    """Return the built-in model profiles in deterministic order.

    :return: Built-in profiles list.
    """
    return list(_BUILTIN_MODEL_PROFILES)


def _builtin_alias_map() -> dict[str, SemanticModelProfile]:
    """Return normalized alias map for built-in profiles.

    :return: Alias-to-profile dictionary.
    """
    alias_map: dict[str, SemanticModelProfile] = {}
    for profile in _BUILTIN_MODEL_PROFILES:
        for alias in profile.all_aliases():
            alias_map[_normalize_model_key(alias)] = profile
    return alias_map


def resolve_local_model_path(model_name: str) -> Path | None:
    """Resolve a model identifier to a local model directory when one exists.

    A model identifier is treated as a local ``save_pretrained``-style directory
    when it points at an existing directory on disk (after ``~`` expansion). Hub
    identifiers like ``org/name`` never resolve here unless a directory of that
    relative name actually exists, mirroring how ``sentence-transformers``
    disambiguates local paths from hub repositories.

    :param model_name: Alias, hub identifier, or filesystem path.
    :return: Resolved absolute directory path, or ``None`` for non-local names.
    """
    candidate = model_name.strip()
    if not candidate:
        return None
    try:
        path = Path(candidate).expanduser()
        if path.is_dir():
            return path.resolve()
    except OSError:
        return None
    return None


def _build_dynamic_gte_modernbert_profile(model_name: str) -> SemanticModelProfile:
    """Build a gte-modernbert-family profile for non-builtin model IDs or local copies.

    :param model_name: Model name or local directory path.
    :return: Dynamic family-appropriate profile.
    """
    builtin = _builtin_alias_map()[_normalize_model_key("gte-modernbert")]
    return SemanticModelProfile(
        key=model_name,
        canonical_name=model_name,
        aliases=(),
        family="gte-modernbert",
        default_semantic_threshold=builtin.default_semantic_threshold,
        default_search_threshold=builtin.default_search_threshold,
    )


def _build_dynamic_embeddinggemma_profile(model_name: str) -> SemanticModelProfile:
    """Build an EmbeddingGemma-family profile for non-builtin model IDs or local copies.

    :param model_name: Model name or local directory path.
    :return: Dynamic family-appropriate profile.
    """
    builtin = _builtin_alias_map()[_normalize_model_key("embeddinggemma")]
    return SemanticModelProfile(
        key=model_name,
        canonical_name=model_name,
        aliases=(),
        family="embeddinggemma",
        default_semantic_threshold=builtin.default_semantic_threshold,
        default_search_threshold=builtin.default_search_threshold,
    )


def resolve_model_profile(model_name: str) -> SemanticModelProfile:
    """Resolve a user model identifier into a concrete model profile.

    Built-in aliases resolve to their profiles. An existing local directory
    (a ``save_pretrained``-style model copy) canonicalizes to its resolved
    absolute path so relative and absolute spellings share one cache identity,
    and its family is inferred from the directory name. Remaining hub-style
    names fall back to name-based family inference.

    :param model_name: Alias, hub model name, or local model directory path.
    :return: Matching profile from builtins or a dynamic fallback.
    """
    alias_map = _builtin_alias_map()
    normalized = _normalize_model_key(model_name)
    builtin = alias_map.get(normalized)
    if builtin is not None:
        return builtin

    local_path = resolve_local_model_path(model_name)
    if local_path is not None:
        canonical = str(local_path)
        family_hint = _normalize_model_key(local_path.name)
    else:
        canonical = model_name
        family_hint = normalized

    if "embeddinggemma" in family_hint:
        return _build_dynamic_embeddinggemma_profile(canonical)
    if "gte-modernbert" in family_hint:
        return _build_dynamic_gte_modernbert_profile(canonical)

    return SemanticModelProfile(
        key=canonical,
        canonical_name=canonical,
        aliases=(),
        family=_GENERIC_PROFILE.family,
        default_semantic_threshold=_GENERIC_PROFILE.default_semantic_threshold,
        default_search_threshold=_GENERIC_PROFILE.default_search_threshold,
    )


def resolve_model_name(model_name: str) -> str:
    """Resolve model name aliases to canonical model IDs.

    :param model_name: Alias or model key.
    :return: Canonical model identifier.
    """
    return resolve_model_profile(model_name).canonical_name


def get_default_semantic_threshold(model_name: str) -> float:
    """Return semantic threshold default for the resolved model profile.

    :param model_name: Alias or model key.
    :return: Default threshold for the resolved profile.
    """
    return resolve_model_profile(model_name).default_semantic_threshold


def get_default_search_threshold(model_name: str) -> float:
    """Return query-search threshold default for the resolved model profile.

    Query-to-code similarity runs far below code-to-code duplicate similarity,
    so search uses a lower floor than duplicate detection.

    :param model_name: Alias or model key.
    :return: Default search threshold for the resolved profile.
    """
    return resolve_model_profile(model_name).default_search_threshold
