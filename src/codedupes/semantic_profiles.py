"""Model profile registry for semantic embedding backends."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from codedupes.constants import HYBRID_STATEMENT_RATIO_MIN, HYBRID_WEAK_JACCARD_MIN

SemanticModelFamily = Literal["gte-modernbert", "embeddinggemma", "generic"]
CalibratedModelFamily = Literal["gte-modernbert", "embeddinggemma"]
ThresholdProfile = Literal["auto", "generic", "embeddinggemma-300m", "gte-modernbert-base"]
THRESHOLD_PROFILE_CHOICES = ("auto", "generic", "embeddinggemma-300m", "gte-modernbert-base")
logger = logging.getLogger(__name__)
_threshold_notice_models: set[tuple[str, SemanticModelFamily]] = set()

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
    # Tier split for semantic-only hybrid pairs: a pair is reported by default
    # (``semantic_high_confidence``) when its identifier Jaccard and statement
    # ratio clear both minimums, or when its similarity clears the language's
    # promotion gate; otherwise it is a withheld ``semantic_review`` candidate.
    # ``None`` (or an absent language) turns similarity promotion off.
    hybrid_weak_identifier_jaccard_min: float = HYBRID_WEAK_JACCARD_MIN
    hybrid_statement_ratio_min: float = HYBRID_STATEMENT_RATIO_MIN
    language_high_confidence_thresholds: Mapping[str, float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze the profile's calibrated thresholds."""
        thresholds = {
            "default_semantic_threshold": self.default_semantic_threshold,
            "default_search_threshold": self.default_search_threshold,
            "hybrid_weak_identifier_jaccard_min": self.hybrid_weak_identifier_jaccard_min,
            "hybrid_statement_ratio_min": self.hybrid_statement_ratio_min,
            **{
                f"language_semantic_thresholds[{language!r}]": threshold
                for language, threshold in self.language_semantic_thresholds.items()
            },
            **{
                f"language_high_confidence_thresholds[{language!r}]": threshold
                for language, threshold in self.language_high_confidence_thresholds.items()
                if threshold is not None
            },
        }
        for name, threshold in thresholds.items():
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be finite and in [0.0, 1.0]")
        for language, threshold in self.language_high_confidence_thresholds.items():
            if threshold is not None and threshold < self.semantic_threshold_for_language(language):
                raise ValueError(
                    f"language_high_confidence_thresholds[{language!r}] must not sit below "
                    "that language's duplicate gate"
                )

        object.__setattr__(
            self,
            "language_semantic_thresholds",
            MappingProxyType(dict(self.language_semantic_thresholds)),
        )
        object.__setattr__(
            self,
            "language_high_confidence_thresholds",
            MappingProxyType(dict(self.language_high_confidence_thresholds)),
        )

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

    def high_confidence_threshold_for_language(self, language: str | None) -> float | None:
        """Return the similarity that promotes an uncorroborated pair to high confidence.

        Promotion widens the default view, so a language without a calibrated
        gate gets none rather than borrowing another language's.

        :param language: Canonical language name, or ``None`` when unknown.
        :return: Calibrated per-language promotion gate, or ``None`` when promotion is off.
        """
        if language is None:
            return None
        return self.language_high_confidence_thresholds.get(language)


# Every builtin profile pins the immutable calibration commit recorded in
# test_fixtures/polyglot_calibration/reports/. Recognized copies also use these
# thresholds as family defaults, without claiming checkpoint equivalence.
# Each per-language duplicate gate
# is the loosest sweep threshold whose F1 stays near that language's best while
# final combined-output precision remains workable (recall-first selection); the profile
# fallback is the strictest calibrated gate and applies only to languages
# without their own calibration entry.
#
# Concretely: a shipped gate is allowed to sit below the sweep's F1-selected
# threshold under two conditions, both bounded by the tested invariant that the
# gate's recall is >= the selection's and its F1 stays within 80% of it
# (tests/test_calibration_reports.py). First, where the sweep shows real recall
# below the F1 pick, the gate follows the recall however many grid steps down
# that is (gte c 0.82 vs 0.90, embeddinggemma javascript 0.72 vs 0.82 and rust
# 0.78 vs 0.82 — each buys measured on-corpus recall). Second, where recall is
# flat, the gate still sits one step loose as an off-corpus generalization
# hedge: the corpora are small, so "no recall gain here" is weak evidence that
# the next repository's near-duplicates sit above the selected threshold, and a
# missed duplicate costs more than an extra review row. Flat-recall loosening
# beyond one step is not taken — the embeddinggemma typescript gate's earlier
# 0.76 (two steps) doubled on-corpus false positives for no measured recall,
# so it was tightened back to 0.78.
#
# The hybrid tier split (which admitted semantic-only pairs the CLI shows by
# default) is swept separately at the shipped gates by
# scripts/sweep_hybrid_gates.py and recorded in
# test_fixtures/polyglot_calibration/reports/corroboration_report.json: the
# corroboration constants are one pooled selection per model that must keep
# >= 85% of published recall and not lower precision in every language, and
# each language's promotion gate is selected the same way at those constants.
# Identifier overlap comes from the same tree-sitter identifier collection in
# every language (Python's includes attribute and keyword-argument names), so
# the identifier floor is a cross-language measurement rather than an artifact
# of one extractor; re-run the sweep after any extraction change and copy every
# regenerated report together. In the recorded sweep no positive identifier
# floor is feasible in every language (the semantic-only positives are
# alpha-renamed, so 0.05 already cuts Rust, TypeScript, and Python recall below
# retention); the statement-ratio floor carries the split for gte-modernbert,
# while for embeddinggemma no size split improves precision without cutting C
# or Rust recall below the floor, so only absurd size mismatches are withheld.
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
        hybrid_weak_identifier_jaccard_min=0.0,
        hybrid_statement_ratio_min=0.80,
        language_high_confidence_thresholds={
            "python": None,
            "c": None,
            "rust": None,
            "javascript": None,
            "typescript": 0.88,
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
            "typescript": 0.78,
        },
        hybrid_weak_identifier_jaccard_min=0.0,
        hybrid_statement_ratio_min=0.20,
        language_high_confidence_thresholds={
            "python": None,
            "c": None,
            "rust": None,
            "javascript": None,
            "typescript": None,
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

    Structured configuration takes precedence over directory and model-card
    hints. This recognizes the family without verifying checkpoint equivalence
    or loading model weights.

    :param model_dir: Resolved local model directory.
    :return: Matching calibrated family, or ``None`` for an unknown model.
    """
    configs: list[dict[str, object]] = []
    for filename in ("config.json", "config_sentence_transformers.json"):
        config_path = model_dir / filename
        try:
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            if (
                filename == "config.json"
                and parsed.get("model_type") == "gemma3_text"
                and parsed.get("use_bidirectional_attention") is True
            ):
                return "embeddinggemma"
            configs.append(parsed)

    for config in configs:
        family = _match_calibrated_family(json.dumps(config))
        if family is not None:
            return family

    path_hints = [model_dir.name]
    path_hints.extend(part for part in model_dir.parts if part.startswith("models--"))
    for hint in path_hints:
        family = _match_calibrated_family(hint)
        if family is not None:
            return family

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


def _build_dynamic_profile(
    model_name: str,
    family: CalibratedModelFamily,
) -> SemanticModelProfile:
    """Build a family-aware profile for a non-builtin model.

    Family thresholds and hybrid tier split are practical defaults for copies
    and fine-tunes, not proof that their score distributions match the
    calibrated checkpoint. The actual model identifier and unpinned revision
    are preserved.

    :param model_name: Model name or local directory path.
    :param family: Built-in family whose loading/prompt behavior applies.
    :return: Dynamic profile with the family's loading behavior and thresholds.
    """
    builtin = next(profile for profile in _BUILTIN_MODEL_PROFILES if profile.family == family)
    return SemanticModelProfile(
        key=model_name,
        canonical_name=model_name,
        aliases=(),
        family=family,
        default_semantic_threshold=builtin.default_semantic_threshold,
        default_search_threshold=builtin.default_search_threshold,
        language_semantic_thresholds=builtin.language_semantic_thresholds,
        hybrid_weak_identifier_jaccard_min=builtin.hybrid_weak_identifier_jaccard_min,
        hybrid_statement_ratio_min=builtin.hybrid_statement_ratio_min,
        language_high_confidence_thresholds=builtin.language_high_confidence_thresholds,
    )


def resolve_model_profile(model_name: str) -> SemanticModelProfile:
    """Resolve a user model identifier into a concrete model profile.

    Built-in aliases resolve to their profiles. An explicit local directory (a
    ``save_pretrained``-style model copy passed as an absolute, dot-relative, or
    home-relative path) canonicalizes to its resolved, true-cased absolute path
    so equivalent path spellings share one cache identity, and its family is
    inferred from configuration, then directory/model-card hints. Remaining
    hub-style names fall back to name-based family inference.

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


def resolve_threshold_profile(
    model_profile: SemanticModelProfile,
    threshold_profile: ThresholdProfile = "auto",
) -> SemanticModelProfile:
    """Select threshold defaults independently of the model used for embeddings.

    :param model_profile: Resolved profile of the actual embedding model.
    :param threshold_profile: ``auto``, ``generic``, or a built-in profile key.
    :return: Profile supplying thresholds only; never use it for model loading.
    :raises ValueError: If the threshold profile is not a supported choice.
    """
    if threshold_profile == "auto":
        return model_profile
    if threshold_profile == "generic":
        return _GENERIC_PROFILE
    for profile in _BUILTIN_MODEL_PROFILES:
        if profile.key == threshold_profile:
            return profile
    raise ValueError(f"threshold_profile must be one of {', '.join(THRESHOLD_PROFILE_CHOICES)}")


def log_family_threshold_notice(profile: SemanticModelProfile) -> None:
    """Explain inferred family defaults once per model when they are selected.

    :param profile: Actual model profile whose family defaults are being used.
    :return: ``None``.
    """
    if profile.default_revision is not None or profile.family == "generic":
        return
    local = is_explicit_local_model_path(profile.canonical_name)
    level = logging.INFO if local else logging.WARNING
    key = (profile.canonical_name, profile.family)
    # Disabled notices must remain available for a later visible run.
    if key in _threshold_notice_models or not logger.isEnabledFor(level):
        return
    _threshold_notice_models.add(key)
    if local:
        logger.info("Use --threshold-profile generic for generic defaults.")
    else:
        logger.warning(
            f"Using {profile.family} family thresholds for {profile.canonical_name}; "
            "this Hub model's score distribution may differ from the calibrated checkpoint. "
            "Use --threshold-profile generic or an explicit numeric threshold to override."
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
