"""Model profile registry for semantic embedding backends."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

SemanticModelFamily = Literal["gte-modernbert", "embeddinggemma", "generic"]
CalibratedModelFamily = Literal["gte-modernbert", "embeddinggemma"]

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
    language_semantic_thresholds: Mapping[str, float] = field(default_factory=dict)

    def all_aliases(self) -> tuple[str, ...]:
        """Return all user-facing names that map to this profile.

        :return: Tuple of alias strings including canonical profile keys.
        """
        return tuple(dict.fromkeys((self.key, self.canonical_name, *self.aliases)))

    def semantic_threshold_for_language(self, language: str | None) -> float:
        """Return the duplicate-detection gate for one canonical language.

        :param language: Canonical language name, or ``None`` when unknown.
        :return: Calibrated per-language gate, or the profile fallback for
            languages without a calibrated entry.
        """
        if language is not None:
            calibrated = self.language_semantic_thresholds.get(language)
            if calibrated is not None:
                return calibrated
        return self.default_semantic_threshold


# Calibrated thresholds are only meaningful against the exact checkpoint they
# were swept on, so every builtin profile pins the immutable commit recorded in
# test_fixtures/polyglot_calibration/reports/. Each per-language duplicate gate
# is the loosest sweep threshold whose F1 stays near that language's best while
# final combined-output precision remains workable (recall-first selection); the profile
# fallback is the strictest calibrated gate and applies only to languages
# without their own calibration entry.
_BUILTIN_MODEL_PROFILES: tuple[SemanticModelProfile, ...] = (
    SemanticModelProfile(
        key="gte-modernbert-base",
        canonical_name="Alibaba-NLP/gte-modernbert-base",
        aliases=(
            "gte-modernbert",
            "alibaba-nlp/gte-modernbert-base",
        ),
        family="gte-modernbert",
        default_revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
        default_semantic_threshold=0.82,
        default_search_threshold=0.50,
        language_semantic_thresholds={
            "python": 0.80,
            "c": 0.82,
            "rust": 0.74,
            "javascript": 0.70,
            "typescript": 0.68,
        },
    ),
    SemanticModelProfile(
        key="embeddinggemma-300m",
        canonical_name="unsloth/embeddinggemma-300m",
        aliases=(
            "google/embeddinggemma-300m",
            "embeddinggemma",
        ),
        family="embeddinggemma",
        default_revision="bfa3c846ac738e62aa61806ef9112d34acb1dc5a",
        default_semantic_threshold=0.78,
        default_search_threshold=0.40,
        language_semantic_thresholds={
            "python": 0.74,
            "c": 0.78,
            "rust": 0.78,
            "javascript": 0.72,
            "typescript": 0.76,
        },
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


def _match_calibrated_family(value: str) -> CalibratedModelFamily | None:
    """Match a model identity hint to a calibrated family.

    :param value: Model name, path component, or serialized metadata.
    :return: Matching calibrated family, or ``None``.
    """
    normalized = _normalize_model_key(value)
    if "embeddinggemma" in normalized:
        return "embeddinggemma"
    if "gte-modernbert" in normalized:
        return "gte-modernbert"
    return None


def _infer_local_model_family(model_dir: Path) -> CalibratedModelFamily | None:
    """Infer a calibrated family from a local model directory.

    The nearest directory name handles intentionally named ``save_pretrained``
    copies. Hugging Face cache snapshots use commit hashes as directory names,
    so their ``models--org--name`` ancestor is also inspected. For arbitrary
    ``hf download --local-dir`` destinations, stable configuration fields and
    the model-card title provide identity without loading model weights.

    :param model_dir: Resolved local model directory.
    :return: Matching calibrated family, or ``None`` for an unknown model.
    """
    path_hints = [model_dir.name]
    path_hints.extend(part for part in model_dir.parts if part.startswith("models--"))
    for hint in path_hints:
        family = _match_calibrated_family(hint)
        if family is not None:
            return family

    config_data: dict[str, object] = {}
    for filename in ("config.json", "config_sentence_transformers.json"):
        config_path = model_dir / filename
        try:
            config_text = config_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        family = _match_calibrated_family(config_text)
        if family is not None:
            return family
        if filename == "config.json":
            try:
                parsed = json.loads(config_text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                config_data = parsed

    if (
        config_data.get("model_type") == "gemma3_text"
        and config_data.get("use_bidirectional_attention") is True
    ):
        return "embeddinggemma"

    try:
        with (model_dir / "README.md").open(encoding="utf-8", errors="replace") as model_card:
            for line_number, line in enumerate(model_card):
                if line.startswith("# "):
                    return _match_calibrated_family(line)
                if line_number >= 127:
                    break
    except OSError:
        pass

    return None


def is_explicit_local_model_path(model_name: str) -> bool:
    """Return whether a model argument unambiguously denotes a filesystem path.

    :param model_name: Model argument from the CLI or Python API.
    :return: ``True`` for absolute, dot-relative, or home-relative paths.
    """
    candidate = model_name.strip()
    return bool(
        candidate
        and (
            Path(candidate).is_absolute()
            or candidate.startswith(("./", "../", "~"))
            or candidate in {".", ".."}
        )
    )


def list_supported_models() -> list[SemanticModelProfile]:
    """Return the built-in model profiles in deterministic order.

    :return: Built-in profiles list.
    """
    return list(_BUILTIN_MODEL_PROFILES)


_BUILTIN_ALIAS_MAP = {
    _normalize_model_key(alias): profile
    for profile in _BUILTIN_MODEL_PROFILES
    for alias in profile.all_aliases()
}


def _true_case_path(path: Path) -> Path:
    """Rebuild an absolute path using each component's on-disk letter case.

    ``Path.resolve()`` follows symlinks but does not normalize letter case, so
    on case-insensitive, case-preserving filesystems (e.g. macOS/APFS) two
    differently-cased spellings of the same directory resolve to two different
    strings. This walks ``path`` component by component from its anchor and
    swaps each one for the exact spelling reported by ``iterdir()``, so the
    result is stable regardless of how the caller spelled it. On genuinely
    case-sensitive filesystems the case-insensitive match degenerates to the
    exact match, so behavior there is unchanged.

    :param path: Absolute, already symlink-resolved path to canonicalize.
    :return: Path with each component corrected to its true on-disk spelling;
        falls back to the remaining resolved components as-is on ``OSError``
        or a missing component.
    """
    true_path = Path(path.anchor)
    remaining = path.relative_to(path.anchor).parts
    for index, part in enumerate(remaining):
        try:
            entries = tuple(entry.name for entry in true_path.iterdir())
        except OSError:
            return true_path.joinpath(*remaining[index:])
        if part in entries:
            true_path /= part
            continue
        match = next((entry for entry in entries if entry.lower() == part.lower()), None)
        true_path /= match if match is not None else part
    return true_path


def resolve_local_model_path(model_name: str) -> Path | None:
    """Resolve a model identifier to a local model directory when one exists.

    Only an absolute, dot-relative, or home-relative argument is treated as a
    ``save_pretrained``-style directory. Requiring explicit path syntax prevents
    a same-named directory in the current working directory from shadowing a
    built-in alias or Hub model ID. The resolved path is additionally true-case
    canonicalized so differently-cased spellings of the same directory on
    case-insensitive filesystems share one cache identity.

    :param model_name: Alias, hub identifier, or filesystem path.
    :return: Resolved absolute directory path, or ``None`` for non-local names.
    """
    candidate = model_name.strip()
    if not candidate or not is_explicit_local_model_path(candidate):
        return None
    try:
        path = Path(candidate).expanduser()
        if path.is_dir():
            return _true_case_path(path.resolve())
    except OSError:
        return None
    return None


@cache
def _warn_uncalibrated_family_copy(model_name: str, family: CalibratedModelFamily) -> None:
    """Warn once that a family-matched model forgoes its family's calibrated gates.

    :param model_name: Canonical model name or local directory path.
    :param family: Built-in family the model was matched to.
    :return: ``None``.
    """
    builtin = next(
        (profile for profile in _BUILTIN_MODEL_PROFILES if profile.family == family), None
    )
    gates = builtin.language_semantic_thresholds if builtin is not None else {}
    gate_text = ", ".join(f"{language}={gate}" for language, gate in gates.items())
    logger.warning(
        f"{model_name} looks like the {family} family but is not the calibrated built-in "
        f"checkpoint, so its per-language duplicate gates ({gate_text}) do not apply. Using the "
        f"uncalibrated generic gate {DEFAULT_FALLBACK_SEMANTIC_THRESHOLD} for every language; "
        "pass an explicit threshold if you calibrated this checkpoint yourself."
    )


def _build_dynamic_profile(
    model_name: str,
    family: CalibratedModelFamily,
) -> SemanticModelProfile:
    """Build a family-aware profile for a non-builtin model.

    Family membership selects loading and prompt behavior only. Calibrated
    thresholds are a property of the exact pinned checkpoint they were swept
    on, and a name or config that merely resembles a family (a fine-tune, a
    modified local copy) can have an entirely different score distribution, so
    dynamic profiles keep the uncalibrated generic thresholds. Pass an explicit
    threshold to override.

    :param model_name: Model name or local directory path.
    :param family: Built-in family whose loading/prompt behavior applies.
    :return: Dynamic family-appropriate profile with generic thresholds.
    """
    _warn_uncalibrated_family_copy(model_name, family)
    return SemanticModelProfile(
        key=model_name,
        canonical_name=model_name,
        aliases=(),
        family=family,
        default_semantic_threshold=_GENERIC_PROFILE.default_semantic_threshold,
        default_search_threshold=_GENERIC_PROFILE.default_search_threshold,
    )


def resolve_model_profile(model_name: str) -> SemanticModelProfile:
    """Resolve a user model identifier into a concrete model profile.

    Built-in aliases resolve to their profiles. An explicit local directory (a
    ``save_pretrained``-style model copy passed as an absolute, dot-relative, or
    home-relative path) canonicalizes to its resolved, true-cased absolute path
    so equivalent path spellings share one cache identity, and its family is
    inferred from that true-cased directory name. Remaining hub-style names fall
    back to name-based family inference.

    :param model_name: Alias, hub model name, or local model directory path.
    :return: Matching profile from builtins or a dynamic fallback.
    """
    normalized = _normalize_model_key(model_name)
    local_path = resolve_local_model_path(model_name)
    if local_path is not None:
        canonical = str(local_path)
        local_family = _infer_local_model_family(local_path)
    else:
        builtin = _BUILTIN_ALIAS_MAP.get(normalized)
        if builtin is not None:
            return builtin
        canonical = model_name
        local_family = None

    family = local_family or _match_calibrated_family(normalized)
    if family is not None:
        return _build_dynamic_profile(canonical, family)

    return SemanticModelProfile(
        key=canonical,
        canonical_name=canonical,
        aliases=(),
        family=_GENERIC_PROFILE.family,
        default_semantic_threshold=_GENERIC_PROFILE.default_semantic_threshold,
        default_search_threshold=_GENERIC_PROFILE.default_search_threshold,
    )


def get_default_semantic_threshold(model_name: str) -> float:
    """Return the fallback semantic duplicate threshold for a model.

    This is the gate for languages without a calibrated per-language entry;
    prefer :func:`get_semantic_threshold_for_language` when the language is
    known.

    :param model_name: Alias or model key.
    :return: Fallback duplicate threshold for the resolved profile.
    """
    return resolve_model_profile(model_name).default_semantic_threshold


def get_semantic_threshold_for_language(model_name: str, language: str | None) -> float:
    """Return the calibrated duplicate gate for a model/language combination.

    :param model_name: Alias or model key.
    :param language: Canonical language name, or ``None`` when unknown.
    :return: Per-language calibrated gate, or the profile fallback.
    """
    return resolve_model_profile(model_name).semantic_threshold_for_language(language)


def get_default_search_threshold(model_name: str) -> float:
    """Return query-search threshold default for the resolved model profile.

    Query-to-code similarity runs far below code-to-code duplicate similarity,
    so search uses a lower floor than duplicate detection.

    :param model_name: Alias or model key.
    :return: Default search threshold for the resolved profile.
    """
    return resolve_model_profile(model_name).default_search_threshold
