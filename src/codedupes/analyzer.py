"""Main analyzer orchestrating all detection methods."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codedupes.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECK_SEMANTIC_TASK,
    DEFAULT_MIN_SEMANTIC_STATEMENTS,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_SEMANTIC_TASK,
    DEFAULT_SEMANTIC_DEVICE,
    DEFAULT_TRADITIONAL_THRESHOLD,
    normalize_semantic_task,
)
from codedupes.devices import normalize_semantic_device, validate_mps_memory_fraction
from codedupes.embedding_cache import capture_cache_warnings, get_embedding_cache
from codedupes.extractor import CodeExtractor
from codedupes.languages.registry import normalize_languages
from codedupes.models import (
    AnalysisResult,
    CodeUnit,
    CodeUnitType,
    DuplicatePair,
    ExtractionDiagnostic,
    HybridDuplicate,
)
from codedupes.pairs import ordered_pair_key
from codedupes.semantic import (
    EmbeddingRunStats,
    EmbeddingSpaceIdentity,
    ProgressMode,
    SearchDocumentMode,
    SemanticBackendError,
    _prepare_search_document,
    embedding_cache_keys_for_units,
    get_code_unit_statement_count,
    get_semantic_runtime_versions,
)
from codedupes.semantic import (
    compute_embeddings_with_identity as compute_embeddings,
)
from codedupes.semantic import (
    run_semantic_analysis_with_identity as run_semantic_analysis,
)
from codedupes.semantic_profiles import resolve_model_profile
from codedupes.traditional import (
    build_reference_graph,
    find_exact_pair_keys,
    find_potentially_unused,
    jaccard_similarity,
    run_traditional_analysis,
    unit_identifier_set,
)

logger = logging.getLogger(__name__)

HYBRID_WEAK_JACCARD_MIN = 0.20
HYBRID_STATEMENT_RATIO_MIN = 0.35
DEFAULT_SEMANTIC_UNIT_TYPES = ("function", "method")
SEMANTIC_UNIT_TYPE_TO_ENUM: dict[str, CodeUnitType] = {
    "function": CodeUnitType.FUNCTION,
    "method": CodeUnitType.METHOD,
    "class": CodeUnitType.CLASS,
}
# Derived so the CLI choice list can never drift from the accepted unit types.
SEMANTIC_UNIT_TYPE_CHOICES: tuple[str, ...] = tuple(SEMANTIC_UNIT_TYPE_TO_ENUM)
DEFAULT_TINY_UNIT_STATEMENT_CUTOFF = 3


def _reject_mode_gated_fields(
    mode_enabled: bool,
    required_flag: str,
    field_checks: tuple[tuple[str, bool], ...],
) -> None:
    """Reject non-default mode-gated fields when their mode is disabled.

    :param mode_enabled: Whether the gating mode is enabled.
    :param required_flag: Config flag name the fields require.
    :param field_checks: Pairs of field name and is-non-default flag.
    :raises ValueError: When any field is non-default while the mode is off.
    """
    if mode_enabled:
        return
    flagged = [name for name, is_set in field_checks if is_set]
    if flagged:
        listed = ", ".join(sorted(flagged))
        raise ValueError(f"{listed} require {required_flag}=True")


def _is_test_function_unit(unit: CodeUnit) -> bool:
    """Return whether the unit is a pytest-style test function.

    Deliberately narrower than the test check in ``find_potentially_unused``
    (no file-name matching, function/method only): this predicate suppresses
    semantic duplicate pairs, where a class or a helper in a ``_test`` file must
    stay eligible for matching.

    :param unit: Code unit under inspection.
    :return: ``True`` for function/method units whose names start with ``test_``.
    """
    return unit.unit_type in {CodeUnitType.FUNCTION, CodeUnitType.METHOD} and unit.name.startswith(
        "test_"
    )


def _statement_count_ratio(unit_a: CodeUnit, unit_b: CodeUnit) -> float:
    """Compute ratio of statement counts for two units.

    :param unit_a: First code unit.
    :param unit_b: Second code unit.
    :return: Ratio of smaller statement count to larger statement count.
    """
    count_a = get_code_unit_statement_count(unit_a)
    count_b = get_code_unit_statement_count(unit_b)
    high = max(count_a, count_b)
    low = min(count_a, count_b)
    if high == 0:
        return 0.0
    return low / high


def _resolve_semantic_unit_type_filter(
    semantic_unit_types: tuple[str, ...],
) -> set[CodeUnitType]:
    """Resolve configured semantic unit type names to enum values.

    :param semantic_unit_types: Configured semantic unit type names.
    :return: Set of comparable enum values.
    """
    return {SEMANTIC_UNIT_TYPE_TO_ENUM[unit_type_name] for unit_type_name in semantic_unit_types}


def _is_tiny_unit(
    unit: CodeUnit,
    statement_cutoff: int,
    statement_cache: dict[str, int],
) -> bool:
    """Return whether a unit is tiny by statement count.

    :param unit: Unit under inspection.
    :param statement_cutoff: Tiny cutoff (exclusive).
    :param statement_cache: Memoized statement counts by unit uid.
    :return: ``True`` when the unit's statement count is below the cutoff.
    """
    count = statement_cache.get(unit.uid)
    if count is None:
        count = get_code_unit_statement_count(unit)
        statement_cache[unit.uid] = count
    return count < statement_cutoff


def _both_units_are_tiny(
    duplicate: DuplicatePair,
    statement_cutoff: int,
    statement_cache: dict[str, int],
    *,
    private_members_included: bool,
) -> bool:
    """Return whether both endpoints are tiny code units.

    :param duplicate: Duplicate pair to inspect.
    :param statement_cutoff: Tiny cutoff (exclusive).
    :param statement_cache: Memoized statement counts by unit uid.
    :param private_members_included: Whether private class members were extracted.
    :return: Whether both endpoints are tiny code units.
    """
    if not private_members_included and (
        duplicate.unit_a.unit_type == CodeUnitType.CLASS
        or duplicate.unit_b.unit_type == CodeUnitType.CLASS
    ):
        return False
    return _is_tiny_unit(duplicate.unit_a, statement_cutoff, statement_cache) and _is_tiny_unit(
        duplicate.unit_b, statement_cutoff, statement_cache
    )


def _tiny_filter_statement_counts(units: list[CodeUnit]) -> dict[str, int]:
    """Measure classes by their extracted members while preserving base counts.

    Class extractors count callable members once because their bodies belong
    to their own code units; static initializer bodies are already counted.
    For tiny-duplicate filtering, expand each member's declaration from one
    statement to that member's effective size so a class containing a few
    substantial methods is not mistaken for a marker.

    :param units: Full extracted analysis scope.
    :return: Effective statement counts keyed by unit uid.
    """
    counts = {unit.uid: get_code_unit_statement_count(unit) for unit in units}
    children: dict[tuple[Path, str], list[CodeUnit]] = {}
    for unit in units:
        parent, separator, _name = unit.qualified_name.rpartition(".")
        if separator:
            children.setdefault((unit.file_path, parent), []).append(unit)

    classes = sorted(
        (unit for unit in units if unit.unit_type == CodeUnitType.CLASS),
        key=lambda unit: unit.qualified_name.count("."),
        reverse=True,
    )
    for unit in classes:
        counts[unit.uid] += sum(
            max(0, counts[member.uid] - 1)
            for member in children.get((unit.file_path, unit.qualified_name), [])
        )
    return counts


def _filter_tiny_traditional_duplicates(
    exact_duplicates: list[DuplicatePair],
    near_duplicates: list[DuplicatePair],
    *,
    units: list[CodeUnit],
    statement_cutoff: int,
    private_members_included: bool,
) -> tuple[list[DuplicatePair], list[DuplicatePair]]:
    """Filter tiny wrapper noise from traditional duplicates.

    :param exact_duplicates: Exact traditional duplicate pairs.
    :param near_duplicates: Near traditional duplicate pairs.
    :param units: Full extracted scope used to measure class members.
    :param statement_cutoff: Tiny cutoff (exclusive).
    :param private_members_included: Whether private class members were extracted.
    :return: Filtered exact and near duplicate lists.
    """
    statement_cache = _tiny_filter_statement_counts(units)
    filtered_exact: list[DuplicatePair] = []
    filtered_near: list[DuplicatePair] = []

    for duplicate in exact_duplicates:
        if _both_units_are_tiny(
            duplicate,
            statement_cutoff,
            statement_cache,
            private_members_included=private_members_included,
        ):
            continue
        filtered_exact.append(duplicate)

    for duplicate in near_duplicates:
        if _both_units_are_tiny(
            duplicate,
            statement_cutoff,
            statement_cache,
            private_members_included=private_members_included,
        ):
            continue
        filtered_near.append(duplicate)

    return filtered_exact, filtered_near


def _synthesize_hybrid_duplicates(
    traditional_duplicates: list[DuplicatePair],
    semantic_duplicates: list[DuplicatePair],
    *,
    jaccard_threshold: float,
    weak_identifier_jaccard_min: float = HYBRID_WEAK_JACCARD_MIN,
    statement_ratio_min: float = HYBRID_STATEMENT_RATIO_MIN,
) -> list[HybridDuplicate]:
    """Build ranked hybrid duplicates from traditional and semantic outputs.

    ``semantic_duplicates`` must already be gated (the pairwise scan applies the
    per-language calibrated gates, or an explicit flat override), so a recorded
    semantic similarity is itself the evidence that the pair cleared its
    duplicate gate. Identifier overlap and statement-count
    similarity promote semantic-only pairs to ``semantic_high_confidence``;
    pairs without that corroboration remain visible as ``semantic_review``.

    :param traditional_duplicates: Traditional duplicate pairs (exact + Jaccard).
    :param semantic_duplicates: Gated semantic duplicate pairs.
    :param jaccard_threshold: Minimum Jaccard similarity used for hybrid tiering.
    :param weak_identifier_jaccard_min: Identifier overlap needed to promote a
        semantic-only candidate to high confidence.
    :param statement_ratio_min: Statement-count ratio needed to promote a
        semantic-only candidate to high confidence.
    :return: Hybrid duplicates sorted by descending confidence. Every candidate
        pair reaches a tier: semantic-only pairs without corroboration fall back
        to ``semantic_review`` rather than being dropped.
    """
    pair_evidence: dict[tuple[str, str], dict[str, object]] = {}

    def ensure_entry(unit_a: CodeUnit, unit_b: CodeUnit) -> dict[str, object]:
        """Return/create a pair evidence map entry.

        :param unit_a: First unit in a candidate pair.
        :param unit_b: Second unit in a candidate pair.
        :return: Shared mutable evidence dict used to combine signals.
        """
        key = ordered_pair_key(unit_a, unit_b)
        entry = pair_evidence.get(key)
        if entry is None:
            entry = {
                "unit_a": unit_a,
                "unit_b": unit_b,
                "has_exact": False,
                "jaccard_similarity": None,
                "semantic_similarity": None,
            }
            pair_evidence[key] = entry
        return entry

    for duplicate in traditional_duplicates:
        entry = ensure_entry(duplicate.unit_a, duplicate.unit_b)
        if duplicate.method in {"ast_hash", "token_hash"}:
            entry["has_exact"] = True
        elif duplicate.method == "jaccard":
            previous = entry["jaccard_similarity"]
            if previous is None or duplicate.similarity > previous:
                entry["jaccard_similarity"] = duplicate.similarity

    for duplicate in semantic_duplicates:
        entry = ensure_entry(duplicate.unit_a, duplicate.unit_b)
        previous = entry["semantic_similarity"]
        if previous is None or duplicate.similarity > previous:
            entry["semantic_similarity"] = duplicate.similarity

    identifier_cache: dict[str, set[str]] = {}
    hybrid_duplicates: list[HybridDuplicate] = []

    for entry in pair_evidence.values():
        unit_a = entry["unit_a"]  # type: ignore[assignment]
        unit_b = entry["unit_b"]  # type: ignore[assignment]
        has_exact = bool(entry["has_exact"])
        jaccard_sim = entry["jaccard_similarity"]  # type: ignore[assignment]
        semantic_sim = entry["semantic_similarity"]  # type: ignore[assignment]

        tier: str | None = None
        confidence: float | None = None
        weak_identifier_jaccard: float | None = None
        statement_ratio: float | None = None

        # Confidence is a corroboration scale, not a raw similarity: at equal
        # evidence strength a tier with more independent corroboration must
        # always outrank one with less, or the weakest tier crowds the
        # best-evidenced pairs off the top of the table. Per tier:
        #   exact                    = 1.0
        #   traditional_near         = 0.55 + 0.45 * jaccard
        #   hybrid_confirmed         = 0.50 * semantic + 0.50 * jaccard
        #   semantic_high_confidence = 0.45 + 0.55 * semantic
        #   semantic_review          = 0.40 + 0.45 * semantic
        # The last two keep semantic_review strictly below its corroborated
        # sibling at every similarity (the gap is 0.05 + 0.10 * semantic).
        if has_exact:
            tier = "exact"
            confidence = 1.0
        elif jaccard_sim is not None and jaccard_sim >= jaccard_threshold:
            if semantic_sim is not None:
                tier = "hybrid_confirmed"
                confidence = (0.5 * semantic_sim) + (0.5 * jaccard_sim)
            else:
                tier = "traditional_near"
                confidence = 0.55 + (0.45 * jaccard_sim)
        elif semantic_sim is not None:
            ids_a = identifier_cache.setdefault(unit_a.uid, unit_identifier_set(unit_a))
            ids_b = identifier_cache.setdefault(unit_b.uid, unit_identifier_set(unit_b))
            weak_identifier_jaccard = jaccard_similarity(ids_a, ids_b)
            statement_ratio = _statement_count_ratio(unit_a, unit_b)

            if (
                weak_identifier_jaccard >= weak_identifier_jaccard_min
                and statement_ratio >= statement_ratio_min
            ):
                tier = "semantic_high_confidence"
                confidence = 0.45 + (0.55 * semantic_sim)
            else:
                tier = "semantic_review"
                confidence = 0.40 + (0.45 * semantic_sim)

        if tier is None or confidence is None:
            continue

        hybrid_duplicates.append(
            HybridDuplicate(
                unit_a=unit_a,
                unit_b=unit_b,
                tier=tier,  # type: ignore[arg-type]
                confidence=float(confidence),
                has_exact=has_exact,
                jaccard_similarity=jaccard_sim,
                semantic_similarity=semantic_sim,
                weak_identifier_jaccard=weak_identifier_jaccard,
                statement_count_ratio=statement_ratio,
            )
        )

    hybrid_duplicates.sort(
        key=lambda duplicate: (
            -duplicate.confidence,
            -(duplicate.semantic_similarity if duplicate.semantic_similarity is not None else -1.0),
            -(duplicate.jaccard_similarity if duplicate.jaccard_similarity is not None else -1.0),
            duplicate.unit_a.uid,
            duplicate.unit_b.uid,
        )
    )

    return hybrid_duplicates


@dataclass
class AnalyzerConfig:
    """Configuration for the code analyzer."""

    # Extraction
    exclude_patterns: list[str] | None = None
    include_private: bool = True
    languages: tuple[str, ...] | None = None

    # Traditional detection
    jaccard_threshold: float = DEFAULT_TRADITIONAL_THRESHOLD

    # Semantic detection
    semantic_threshold: float | None = None
    cross_language: bool = False
    model_name: str = DEFAULT_MODEL
    semantic_task: str | None = None
    instruction_prefix: str | None = None
    model_revision: str | None = None
    trust_remote_code: bool | None = None
    device: str = DEFAULT_SEMANTIC_DEVICE
    mps_fallback: bool | None = None
    mps_memory_fraction: float | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    min_semantic_statements: int = DEFAULT_MIN_SEMANTIC_STATEMENTS
    semantic_unit_types: tuple[str, ...] = DEFAULT_SEMANTIC_UNIT_TYPES
    include_stubs: bool = False
    allow_semantic_fallback: bool = False
    filter_tiny_traditional: bool = True
    tiny_unit_statement_cutoff: int = DEFAULT_TINY_UNIT_STATEMENT_CUTOFF
    embedding_cache: bool = True
    strict_revision_cache: bool = False
    progress: ProgressMode = "auto"
    search_document: SearchDocumentMode = "source"

    # What to run. mode="check" validates the calibrated duplicate-gate
    # contract at construction; mode="search" defers to the query-time search
    # contract instead, because search calibration is independent of the
    # duplicate gates. search()/index() accept both; analyze() requires "check".
    mode: str = "check"
    run_traditional: bool = True
    run_semantic: bool = True
    run_unused: bool = True
    strict_unused: bool = False
    suppress_test_semantic_matches: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"check", "search"}:
            raise ValueError(f"mode must be 'check' or 'search', got {self.mode!r}")
        if self.mode == "search" and not self.run_semantic:
            raise ValueError("mode='search' requires run_semantic=True")
        if self.search_document not in {"source", "contextual"}:
            raise ValueError("search_document must be 'source' or 'contextual'")
        if self.progress not in {"auto", "always", "never"}:
            raise ValueError("progress must be 'auto', 'always', or 'never'")

        self.languages = normalize_languages(self.languages)

        if not 0.0 <= self.jaccard_threshold <= 1.0:
            raise ValueError("jaccard_threshold must be in [0.0, 1.0]")

        if self.semantic_threshold is not None and not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be in [0.0, 1.0]")

        self.device = normalize_semantic_device(self.device)
        self.mps_memory_fraction = validate_mps_memory_fraction(self.mps_memory_fraction)

        if self.mps_memory_fraction is not None and self.device not in {"auto", "mps"}:
            raise ValueError("mps_memory_fraction requires device='mps' or device='auto'")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if self.min_semantic_statements < 0:
            raise ValueError("min_semantic_statements must be >= 0")

        if not self.semantic_unit_types:
            raise ValueError("semantic_unit_types must contain at least one unit type")
        normalized_types = tuple(
            unit_type.strip().lower() for unit_type in self.semantic_unit_types
        )
        invalid_types = sorted(
            unit_type
            for unit_type in normalized_types
            if unit_type not in SEMANTIC_UNIT_TYPE_TO_ENUM
        )
        if invalid_types:
            allowed = ", ".join(sorted(SEMANTIC_UNIT_TYPE_TO_ENUM))
            invalid = ", ".join(invalid_types)
            raise ValueError(f"Invalid semantic_unit_types: {invalid}. Allowed values: {allowed}")
        self.semantic_unit_types = tuple(dict.fromkeys(normalized_types))

        if self.tiny_unit_statement_cutoff < 0:
            raise ValueError("tiny_unit_statement_cutoff must be >= 0")

        if self.semantic_task is not None:
            self.semantic_task = normalize_semantic_task(
                self.semantic_task,
                default_task=DEFAULT_CHECK_SEMANTIC_TASK,
            )

        if not self.run_unused and self.strict_unused:
            raise ValueError("strict_unused requires run_unused=True")

        _reject_mode_gated_fields(
            self.run_semantic,
            "run_semantic",
            (
                ("semantic_threshold", self.semantic_threshold is not None),
                ("cross_language", self.cross_language),
                ("semantic_task", self.semantic_task is not None),
                ("instruction_prefix", self.instruction_prefix is not None),
                ("model_revision", self.model_revision is not None),
                ("trust_remote_code", self.trust_remote_code is not None),
                ("device", self.device != DEFAULT_SEMANTIC_DEVICE),
                ("mps_fallback", self.mps_fallback is not None),
                ("mps_memory_fraction", self.mps_memory_fraction is not None),
                ("strict_revision_cache", self.strict_revision_cache),
                ("batch_size", self.batch_size != DEFAULT_BATCH_SIZE),
                ("suppress_test_semantic_matches", self.suppress_test_semantic_matches),
            ),
        )

        _reject_mode_gated_fields(
            self.run_traditional,
            "run_traditional",
            (
                ("jaccard_threshold", self.jaccard_threshold != DEFAULT_TRADITIONAL_THRESHOLD),
                ("filter_tiny_traditional", not self.filter_tiny_traditional),
                (
                    "tiny_unit_statement_cutoff",
                    self.tiny_unit_statement_cutoff != DEFAULT_TINY_UNIT_STATEMENT_CUTOFF,
                ),
            ),
        )

        if self.allow_semantic_fallback and (not self.run_semantic or not self.run_traditional):
            raise ValueError(
                "allow_semantic_fallback requires run_semantic=True and run_traditional=True"
            )

        if self.mode == "check" and self.run_semantic and self.semantic_threshold is None:
            reasons = self._uncalibrated_gate_reasons(
                self.semantic_task or DEFAULT_CHECK_SEMANTIC_TASK
            )
            if reasons:
                context = ", ".join(reasons)
                raise ValueError(
                    f"The default duplicate thresholds are not calibrated for {context}; "
                    "provide semantic_threshold explicitly."
                )

    def _uncalibrated_gate_reasons(self, semantic_task: str) -> list[str]:
        """Collect config choices the calibrated duplicate gates do not cover.

        :param semantic_task: Resolved task used to embed duplicate candidates.
        :return: Human-readable reasons, empty when the calibrated gates apply.
        """
        profile = resolve_model_profile(self.model_name)
        reasons: list[str] = []
        if self.instruction_prefix is not None:
            reasons.append("a custom instruction prefix")
        if profile.family == "embeddinggemma" and semantic_task != DEFAULT_CHECK_SEMANTIC_TASK:
            reasons.append(f"semantic task {semantic_task!r}")
        if (
            profile.default_revision is not None
            and self.model_revision is not None
            and self.model_revision != profile.default_revision
        ):
            reasons.append(f"model revision {self.model_revision!r}")
        # Remote-code execution splits the embedding cache key because it can
        # change the vectors, so it must invalidate the calibrated gates too.
        if (
            self.trust_remote_code is not None
            and self.trust_remote_code != profile.default_trust_remote_code
        ):
            reasons.append(f"trust_remote_code={self.trust_remote_code}")
        return reasons


class CodeAnalyzer:
    """
    Main analyzer for detecting duplicate and unused code.

    Combines structural/token methods with semantic embedding similarity.
    """

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        """Initialize analyzer state.

        :param config: Optional analyzer configuration override.
        """
        self.config = config or AnalyzerConfig()
        self._units: list[CodeUnit] | None = None
        self._embeddings: np.ndarray | None = None
        self._semantic_units: list[CodeUnit] | None = None
        self._resolved_search_semantic_task: str | None = None
        self._embedding_space_identity: EmbeddingSpaceIdentity | None = None
        self._embedding_stats: EmbeddingRunStats | None = None
        self._cache_scope: Path | None = None
        self._extraction_diagnostics: list[ExtractionDiagnostic] = []
        self._semantic_diagnostics: list[ExtractionDiagnostic] = []

    @property
    def semantic_diagnostics(self) -> list[ExtractionDiagnostic]:
        """Return diagnostics raised by the semantic stage of the last run.

        :return: Diagnostics for units excluded from semantic comparison.
        """
        return list(self._semantic_diagnostics)

    @property
    def extracted_unit_count(self) -> int:
        """Return the number of code units extracted by the last run.

        This count precedes semantic candidate filtering, so callers can
        distinguish an empty source corpus from an empty semantic index.

        :return: Extracted code-unit count, or zero before the first run.
        """
        return len(self._units) if self._units is not None else 0

    @property
    def embedding_stats(self) -> EmbeddingRunStats | None:
        """Return telemetry from the most recent successful semantic corpus run."""
        return self._embedding_stats

    def _reset_analysis_state(self, cache_scope: Path) -> None:
        """Clear corpus-specific state before one analysis run.

        :param cache_scope: Root path used to address embedding-cache entries.
        :return: ``None``.
        """
        self._units = None
        self._embeddings = None
        self._semantic_units = None
        self._resolved_search_semantic_task = None
        self._embedding_space_identity = None
        self._embedding_stats = None
        self._cache_scope = cache_scope
        self._extraction_diagnostics = []
        self._semantic_diagnostics = []

    def _publish_corpus_manifest(
        self,
        path: Path,
        semantic_units: list[CodeUnit],
        semantic_task: str,
        *,
        search_document: SearchDocumentMode = "source",
        document_texts: list[str] | None = None,
    ) -> None:
        """Publish cache corpus metadata after a successful analyzer run.

        :param path: Analysis target, preserving an explicit symlink's name.
        :param semantic_units: Units selected for semantic embedding.
        :param semantic_task: Effective embedding task.
        :param search_document: Search document representation used by the run.
        :param document_texts: Optional contextual texts aligned with ``semantic_units``.
        :return: ``None``.
        """
        # Missing units are not authoritative deletions when extraction could
        # not observe the requested source, including explicit file targets.
        if any(
            diagnostic.code
            in {
                "read-error",
                "parse-error",
                "invalid-utf8",
                "partial-parse",
                "unit-parse-error",
                "walk-error",
            }
            for diagnostic in self._extraction_diagnostics
        ):
            return
        stats = self._embedding_stats
        identity = self._embedding_space_identity
        if (
            stats is None
            or not stats.cache_enabled
            or stats.cache_revision is None
            or identity is None
            or self._cache_scope is None
        ):
            return
        with capture_cache_warnings(stats.cache_warnings):
            cache = get_embedding_cache()
        if cache is None:
            return
        selection_payload = {
            "include_private": self.config.include_private,
            "languages": self.config.languages,
            "min_semantic_statements": self.config.min_semantic_statements,
            "semantic_unit_types": self.config.semantic_unit_types,
            "include_stubs": self.config.include_stubs,
            "exclude_patterns": self.config.exclude_patterns,
            "semantic_task": semantic_task,
            "instruction_prefix": self.config.instruction_prefix,
            "runtime_variant": identity.runtime_variant,
            "search_document": search_document,
        }
        selection = hashlib.blake2b(
            json.dumps(selection_payload, sort_keys=True).encode(),
            digest_size=16,
        ).hexdigest()
        unit_keys = embedding_cache_keys_for_units(
            semantic_units,
            identity,
            revision=stats.cache_revision,
            document_texts=document_texts,
            search_document=search_document,
        )
        # Use extraction's file identity when replacing a single-file slice,
        # including when no semantic candidates remain after the rescan.
        observed_path = path.resolve()
        if not observed_path.is_relative_to(self._cache_scope):
            observed_path = path
        with capture_cache_warnings(stats.cache_warnings):
            published = cache.publish_corpus_manifest(
                self._cache_scope,
                identity.model_name,
                stats.cache_revision,
                selection=selection,
                units=unit_keys,
                # Excludes belong to the selection digest; a successful walk
                # fully observes that selection even when it omits files.
                complete_scan=path.is_dir(),
                unit_paths={unit.uid: str(unit.file_path) for unit in semantic_units},
                observed_files=(str(observed_path),) if path.is_file() else (),
            )
        if published is None:
            return
        stats.moved_units_reused = len(published.diff.moved)
        stats.deleted_units = len(published.diff.deleted)
        stats.orphan_rows_retained = published.orphan_rows_retained
        stats.orphan_rows_collected = published.orphan_rows_collected
        stats.manifest_generation = published.generation

    def _extract_corpus_units(self, path: Path) -> list[CodeUnit]:
        """Extract code units while preserving an explicit file target's name.

        :param path: Existing directory or file path with a resolved parent.
        :return: Extracted code units.
        """
        logger.info(f"Extracting code units from {path}")

        if path.is_file():
            extractor = CodeExtractor(
                path.parent,
                exclude_patterns=self.config.exclude_patterns,
                include_private=self.config.include_private,
                # include_stubs gates directory walks; a .pyi named explicitly
                # as the analysis target is analyzed as given.
                include_stubs=self.config.include_stubs or path.suffix.lower() == ".pyi",
                languages=self.config.languages,
            )
            units = list(extractor.extract_from_file(path))
        else:
            extractor = CodeExtractor(
                path,
                exclude_patterns=self.config.exclude_patterns,
                include_private=self.config.include_private,
                include_stubs=self.config.include_stubs,
                languages=self.config.languages,
            )
            units = extractor.extract_all()

        self._extraction_diagnostics = list(extractor.diagnostics)
        logger.info(f"Extracted {len(units)} code units")
        return units

    def _select_semantic_candidates(self, units: list[CodeUnit]) -> list[CodeUnit]:
        """Filter units eligible for semantic embedding by type and statement count.

        :param units: Extracted code units.
        :return: Units passing the semantic type filter and statement-count gate.
        """
        semantic_type_filter = _resolve_semantic_unit_type_filter(self.config.semantic_unit_types)
        return [
            unit
            for unit in units
            if unit.unit_type in semantic_type_filter
            and get_code_unit_statement_count(unit) >= self.config.min_semantic_statements
        ]

    def _resolve_semantic_gates(
        self,
        semantic_candidates: list[CodeUnit],
        semantic_task: str,
    ) -> tuple[dict[str, float], float]:
        """Resolve per-language duplicate gates and the scan fallback floor.

        An explicit ``config.semantic_threshold`` applies as one flat gate to
        every language. Otherwise each candidate language gets its calibrated
        gate from the model profile, and the pairwise scan holds every group to
        its own gate. The returned floor is the loosest gate, used only for a
        language that reaches the scan without a calibrated entry.

        :param semantic_candidates: Units eligible for semantic comparison.
        :param semantic_task: Resolved task used to embed the candidates.
        :return: Tuple of the per-language gate map and the fallback floor.
        :raises ValueError: If the configured embedding context has no calibrated
            default threshold.
        """
        explicit = self.config.semantic_threshold
        languages = sorted({unit.language for unit in semantic_candidates})
        if explicit is not None:
            return dict.fromkeys(languages, explicit), explicit

        profile = resolve_model_profile(self.config.model_name)
        # Backstop for configs mutated after construction; __post_init__
        # enforces this for every check-mode config it accepts.
        uncalibrated_reasons = self.config._uncalibrated_gate_reasons(semantic_task)
        if uncalibrated_reasons:
            context = ", ".join(uncalibrated_reasons)
            raise ValueError(
                f"The default duplicate thresholds are not calibrated for {context}; "
                "provide semantic_threshold explicitly."
            )

        gates = {
            language: profile.semantic_threshold_for_language(language) for language in languages
        }
        floor = min(gates.values(), default=profile.semantic_threshold_for_language(None))
        if gates:
            gate_text = ", ".join(f"{language}={gate:.2f}" for language, gate in gates.items())
            logger.info(
                f"Per-language semantic duplicate gates: {gate_text} "
                f"(fallback {floor:.2f} for uncalibrated languages)"
            )
        return gates, floor

    def analyze(self, path: Path | str) -> AnalysisResult:
        """
        Run full analysis on a directory or file.

        Args:
            path: Path to a supported source file or directory

        Returns:
            AnalysisResult with all findings

        Raises:
            FileNotFoundError: If path does not exist
            ValueError: If the config was built with mode="search"
            SemanticBackendError: If semantic-only analysis cannot use the requested device
            RuntimeError: If combined analysis cannot use the semantic backend and fallback
                is disabled
        """
        if self.config.mode != "check":
            raise ValueError(
                "analyze() requires a mode='check' config; mode='search' configs "
                "skip duplicate-gate validation and only support index()/search()."
            )
        path = Path(path)
        path = path.parent.resolve() / path.name if path.is_file() else path.resolve()

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        self._reset_analysis_state(path.parent if path.is_file() else path)

        units = self._extract_corpus_units(path)
        self._units = units

        traditional_duplicates: list[DuplicatePair] = []
        unused: list[CodeUnit] = []
        semantic_fallback = False
        semantic_fallback_reason: str | None = None
        semantic_task = self.config.semantic_task or DEFAULT_CHECK_SEMANTIC_TASK
        self._resolved_search_semantic_task = semantic_task

        semantic_candidates: list[CodeUnit] = []
        semantic_gates: dict[str, float] = {}
        semantic_scan_floor = 0.0
        if self.config.run_semantic:
            semantic_candidates = self._select_semantic_candidates(units)
            self._semantic_units = semantic_candidates
            semantic_gates, semantic_scan_floor = self._resolve_semantic_gates(
                semantic_candidates,
                semantic_task,
            )

        if self.config.run_traditional:
            exact_dupes, near_dupes, _ = run_traditional_analysis(
                units,
                jaccard_threshold=self.config.jaccard_threshold,
                compute_unused=False,
            )
            if self.config.filter_tiny_traditional:
                exact_dupes, near_dupes = _filter_tiny_traditional_duplicates(
                    exact_dupes,
                    near_dupes,
                    units=units,
                    statement_cutoff=self.config.tiny_unit_statement_cutoff,
                    private_members_included=self.config.include_private,
                )
            traditional_duplicates = exact_dupes + near_dupes

        unused_excluded_units = 0

        semantic_duplicates: list[DuplicatePair] = []
        embedding_stats: EmbeddingRunStats | None = None

        if self.config.run_semantic:
            embedding_stats = EmbeddingRunStats()
            exclude: set[tuple[str, str]] = set()

            if self.config.run_traditional:
                # Exclude every exact-hash pair — including pairs the tiny filter stripped
                # from traditional output — so semantic scoring can never re-report an
                # exact duplicate as a new lower-confidence match. Near-duplicate (jaccard)
                # pairs stay out of exclusion so semantic scoring can confirm traditional
                # evidence and enable hybrid_confirmed scoring.
                exclude = find_exact_pair_keys(semantic_candidates)

            try:
                semantic_kwargs: dict[str, object] = {
                    "model_name": self.config.model_name,
                    "instruction_prefix": self.config.instruction_prefix,
                    "threshold": semantic_scan_floor,
                    "language_thresholds": semantic_gates,
                    "exclude_pairs": exclude,
                    "batch_size": self.config.batch_size,
                    "revision": self.config.model_revision,
                    "trust_remote_code": self.config.trust_remote_code,
                    "semantic_task": semantic_task,
                    "device": self.config.device,
                    "mps_fallback": self.config.mps_fallback,
                    "mps_memory_fraction": self.config.mps_memory_fraction,
                    "use_cache": self.config.embedding_cache,
                    "cache_scope": self._cache_scope,
                    "strict_revision_cache": self.config.strict_revision_cache,
                    "cross_language": self.config.cross_language,
                    "progress": self.config.progress,
                    "stats": embedding_stats,
                }
                (
                    self._embeddings,
                    semantic_duplicates,
                    self._embedding_space_identity,
                ) = run_semantic_analysis(semantic_candidates, **semantic_kwargs)
            except (ModuleNotFoundError, SemanticBackendError, RuntimeError) as exc:
                self._embedding_space_identity = None
                embedding_stats = None
                # If semantic is the only duplicate-detection method requested,
                # fail hard instead of silently degrading to unused-only output.
                if not self.config.run_traditional:
                    raise
                if not self.config.allow_semantic_fallback:
                    raise RuntimeError(
                        f"Semantic analysis failed in combined mode ({exc}). Re-run with "
                        "`--allow-semantic-fallback` to keep full-scope traditional results, "
                        "or use `--traditional-only` for deterministic non-semantic analysis."
                    ) from exc
                semantic_fallback = True
                self._embeddings = None
                semantic_duplicates = []
                runtime_versions = get_semantic_runtime_versions()
                version_text = ", ".join(
                    f"{key}={value}" for key, value in runtime_versions.items()
                )
                semantic_fallback_reason = (
                    f"Semantic analysis unavailable ({exc}). Proceeding with non-semantic "
                    "analysis on the full traditional scope "
                    f"(allow_semantic_fallback=True). model={self.config.model_name} "
                    f"revision={self.config.model_revision} "
                    f"trust_remote_code={self.config.trust_remote_code} "
                    f"device={self.config.device} "
                    f"mps_fallback={self.config.mps_fallback} "
                    f"mps_memory_fraction={self.config.mps_memory_fraction} "
                    f"[{version_text}]. "
                    f"Retry with `codedupes check {path} --traditional-only`."
                )
                logger.warning(semantic_fallback_reason)
            else:
                self._embedding_stats = embedding_stats

            # Language partitioning and the per-language gates are applied inside
            # the pairwise scan (see find_semantic_duplicates), so every pair that
            # arrives here already cleared its own language's gate.

            if self.config.suppress_test_semantic_matches:
                semantic_duplicates = [
                    duplicate
                    for duplicate in semantic_duplicates
                    if not (
                        _is_test_function_unit(duplicate.unit_a)
                        or _is_test_function_unit(duplicate.unit_b)
                    )
                ]

        if self.config.run_unused:
            build_reference_graph(units, project_root=path)
            unused = find_potentially_unused(units, strict_unused=self.config.strict_unused)
            unused_excluded_units = sum(unit.language != "python" for unit in units)
            logger.info(f"Found {len(unused)} potentially unused code units")

            if self.config.run_semantic:
                unused_uids = {unit.uid for unit in unused}
                semantic_duplicates = [
                    duplicate
                    for duplicate in semantic_duplicates
                    if not (
                        duplicate.unit_a.uid in unused_uids and duplicate.unit_b.uid in unused_uids
                    )
                ]

        combined_mode = self.config.run_traditional and self.config.run_semantic
        hybrid_duplicates: list[HybridDuplicate] = []

        if combined_mode:
            hybrid_duplicates = _synthesize_hybrid_duplicates(
                traditional_duplicates,
                semantic_duplicates,
                jaccard_threshold=self.config.jaccard_threshold,
            )

        if not units:
            analysis_mode = "none"
        elif combined_mode:
            analysis_mode = "combined"
        elif self.config.run_traditional:
            analysis_mode = "traditional"
        elif self.config.run_semantic:
            analysis_mode = "semantic"
        else:
            analysis_mode = "none"

        if embedding_stats is not None:
            self._publish_corpus_manifest(path, semantic_candidates, semantic_task)

        return AnalysisResult(
            units=units,
            traditional_duplicates=traditional_duplicates,
            semantic_duplicates=semantic_duplicates,
            hybrid_duplicates=hybrid_duplicates,
            potentially_unused=unused,
            analysis_mode=analysis_mode,
            semantic_fallback=semantic_fallback,
            semantic_fallback_reason=semantic_fallback_reason,
            extraction_diagnostics=list(self._extraction_diagnostics),
            semantic_diagnostics=list(self._semantic_diagnostics),
            unused_excluded_units=unused_excluded_units,
            embedding_stats=embedding_stats,
        )

    def index(self, path: Path | str) -> int:
        """Build the semantic search corpus without mining duplicate pairs.

        Extracts code units, filters semantic candidates, and computes (or
        loads from cache) their embeddings so :meth:`search` can run. Unlike
        :meth:`analyze`, no all-pairs duplicate scan, traditional analysis, or
        unused-code analysis happens, so indexing stays linear in corpus size.

        Inputs are passed to the embedding backend unchanged. Models apply their
        own normal context-window truncation to both corpus units and queries.

        :param path: Path to a supported source file or directory.
        :return: Number of code units embedded for search.
        :raises FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        path = path.parent.resolve() / path.name if path.is_file() else path.resolve()

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        self._reset_analysis_state(path.parent if path.is_file() else path)

        units = self._extract_corpus_units(path)
        self._units = units
        self._resolved_search_semantic_task = (
            self.config.semantic_task or DEFAULT_SEARCH_SEMANTIC_TASK
        )
        semantic_candidates = self._select_semantic_candidates(units)
        self._semantic_units = semantic_candidates
        self._embedding_stats = EmbeddingRunStats()
        document_texts = (
            [_prepare_search_document(unit, self._cache_scope) for unit in semantic_candidates]
            if self.config.search_document == "contextual" and self._cache_scope is not None
            else None
        )

        try:
            self._embeddings, self._embedding_space_identity = compute_embeddings(
                semantic_candidates,
                model_name=self.config.model_name,
                instruction_prefix=self.config.instruction_prefix,
                batch_size=self.config.batch_size,
                revision=self.config.model_revision,
                trust_remote_code=self.config.trust_remote_code,
                semantic_task=self._resolved_search_semantic_task,
                device=self.config.device,
                mps_fallback=self.config.mps_fallback,
                mps_memory_fraction=self.config.mps_memory_fraction,
                use_cache=self.config.embedding_cache,
                cache_scope=self._cache_scope,
                strict_revision_cache=self.config.strict_revision_cache,
                progress=self.config.progress,
                stats=self._embedding_stats,
                document_texts=document_texts,
                search_document=self.config.search_document,
            )
        except Exception:
            self._embedding_space_identity = None
            self._embedding_stats = None
            raise
        self._publish_corpus_manifest(
            path,
            semantic_candidates,
            self._resolved_search_semantic_task or DEFAULT_SEARCH_SEMANTIC_TASK,
            search_document=self.config.search_document,
            document_texts=document_texts,
        )
        return len(semantic_candidates)

    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float | None = None,
    ) -> list[tuple[CodeUnit, float]]:
        """
        Search for code units matching a natural language query.

        Must run index() (or analyze() with semantic analysis enabled) first to
        compute embeddings. The search floor is ``threshold`` when given, else
        ``config.semantic_threshold``, else the model profile's search default
        (far looser than a duplicate gate, because query-to-code similarity runs
        well below code-to-code similarity). Prefer ``threshold`` over setting
        ``config.semantic_threshold``: the latter also replaces every calibrated
        per-language duplicate gate with one flat value.

        Searching after :meth:`analyze` reuses the duplicate-detection task, for
        which no search default is calibrated on prompt-sensitive models, so
        those combinations require ``threshold``.

        Contextual indexes require an explicit per-call or configured threshold
        because the source-only search default is not calibrated for their input.

        :param query: Search query string.
        :param top_k: Maximum results to return.
        :param threshold: Finite minimum cosine similarity for this call only;
            negative floors are allowed.
        :return: List of code units and cosine scores.
        :raises ValueError: If ``threshold`` is non-finite, or the corpus has no
            calibrated search default and neither ``threshold`` nor
            ``config.semantic_threshold`` is supplied.
        """
        if self._units is None or self._embeddings is None:
            raise RuntimeError(
                "Must run index() or analyze() with run_semantic=True before search()."
            )

        resolved_threshold = threshold if threshold is not None else self.config.semantic_threshold
        from codedupes.semantic import find_similar_to_query

        if self._resolved_search_semantic_task is None:
            raise RuntimeError("Semantic configuration was not resolved; run analyze() first.")

        warning_collector = (
            self._embedding_stats.cache_warnings if self._embedding_stats is not None else None
        )
        with capture_cache_warnings(warning_collector):
            return find_similar_to_query(
                query,
                self._semantic_units or [],
                self._embeddings,
                model_name=self.config.model_name,
                instruction_prefix=self.config.instruction_prefix,
                top_k=top_k,
                revision=self.config.model_revision,
                trust_remote_code=self.config.trust_remote_code,
                threshold=resolved_threshold,
                semantic_task=self._resolved_search_semantic_task,
                device=self.config.device,
                mps_fallback=self.config.mps_fallback,
                mps_memory_fraction=self.config.mps_memory_fraction,
                use_cache=self.config.embedding_cache,
                cache_scope=self._cache_scope,
                corpus_identity=self._embedding_space_identity,
                strict_revision_cache=self.config.strict_revision_cache,
            )


def analyze_directory(
    path: Path | str,
    semantic_threshold: float | None = None,
    cross_language: bool = False,
    traditional_threshold: float = DEFAULT_TRADITIONAL_THRESHOLD,
    exclude_patterns: list[str] | None = None,
    languages: tuple[str, ...] | None = None,
    model_name: str = DEFAULT_MODEL,
    semantic_task: str | None = None,
    instruction_prefix: str | None = None,
    model_revision: str | None = None,
    trust_remote_code: bool | None = None,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    mps_fallback: bool | None = None,
    mps_memory_fraction: float | None = None,
    min_semantic_statements: int = DEFAULT_MIN_SEMANTIC_STATEMENTS,
    semantic_unit_types: tuple[str, ...] = DEFAULT_SEMANTIC_UNIT_TYPES,
    filter_tiny_traditional: bool = True,
    tiny_unit_statement_cutoff: int = DEFAULT_TINY_UNIT_STATEMENT_CUTOFF,
    include_stubs: bool = False,
    allow_semantic_fallback: bool = False,
    run_unused: bool = True,
    strict_unused: bool = False,
) -> AnalysisResult:
    """
    Convenience function for quick analysis.

    Args:
        path: Directory to analyze
        semantic_threshold: Flat cosine gate applied to every language; ``None``
            uses the model profile's calibrated per-language gates
        cross_language: Report semantic duplicate pairs across languages
            (uncalibrated; a mixed pair uses the looser of its two gates)
        traditional_threshold: Jaccard threshold for traditional near-duplicates
        exclude_patterns: Glob patterns for files to exclude
        languages: Optional language filter; omitted means auto-detect supported files.
        model_name: HuggingFace model for embeddings
        semantic_task: Semantic task mode for prompt/inference behavior
        instruction_prefix: Custom instruction prefix prepended to semantic inputs
        model_revision: Optional HuggingFace model revision/commit hash.
            If None, semantic backend chooses model-specific default behavior.
        trust_remote_code: Whether remote model code may execute while loading.
        device: Semantic inference device: ``auto``, ``cpu``, ``cuda``, or ``mps``.
        mps_fallback: Whether unsupported MPS operators may fall back to CPU.
            ``None`` enables the safe automatic policy while respecting an existing
            ``PYTORCH_ENABLE_MPS_FALLBACK`` environment setting.
        mps_memory_fraction: Optional PyTorch MPS allocator fraction in ``(0, 2]``.
        min_semantic_statements: Minimum statement count required for semantic analysis.
        semantic_unit_types: Unit types eligible for semantic embeddings.
        filter_tiny_traditional: Filter tiny traditional duplicates when true.
        tiny_unit_statement_cutoff: Tiny code-unit cutoff (exclusive).
        include_stubs: Whether to analyze ``.pyi`` files.
        allow_semantic_fallback: Allow combined mode to keep full-scope traditional results
            when semantic backend loading/inference fails.
        strict_unused: Whether to ignore public API exclusions when reporting unused code.
        run_unused: Run potentially-unused detection even when traditional analysis is off

    Returns:
        AnalysisResult
    """
    config = AnalyzerConfig(
        semantic_threshold=semantic_threshold,
        cross_language=cross_language,
        jaccard_threshold=traditional_threshold,
        exclude_patterns=exclude_patterns,
        languages=languages,
        model_name=model_name,
        semantic_task=semantic_task,
        instruction_prefix=instruction_prefix,
        model_revision=model_revision,
        trust_remote_code=trust_remote_code,
        device=device,
        mps_fallback=mps_fallback,
        mps_memory_fraction=mps_memory_fraction,
        min_semantic_statements=min_semantic_statements,
        semantic_unit_types=semantic_unit_types,
        filter_tiny_traditional=filter_tiny_traditional,
        tiny_unit_statement_cutoff=tiny_unit_statement_cutoff,
        include_stubs=include_stubs,
        allow_semantic_fallback=allow_semantic_fallback,
        run_unused=run_unused,
        strict_unused=strict_unused,
    )

    analyzer = CodeAnalyzer(config)
    return analyzer.analyze(path)
