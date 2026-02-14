"""Main analyzer orchestrating all detection methods."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codedupes.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECK_SEMANTIC_TASK,
    DEFAULT_SEARCH_SEMANTIC_TASK,
    DEFAULT_MIN_SEMANTIC_LINES,
    DEFAULT_MODEL,
    DEFAULT_TRADITIONAL_THRESHOLD,
)
from codedupes.extractor import CodeExtractor
from codedupes.models import AnalysisResult, CodeUnit, CodeUnitType, DuplicatePair, HybridDuplicate
from codedupes.pairs import ordered_pair_key
from codedupes.semantic import (
    SemanticBackendError,
    get_code_unit_statement_count,
    get_semantic_runtime_versions,
    run_semantic_analysis,
)
from codedupes.semantic_profiles import get_default_semantic_threshold
from codedupes.traditional import (
    build_reference_graph,
    extract_identifiers,
    find_potentially_unused,
    jaccard_similarity,
    run_traditional_analysis,
)

logger = logging.getLogger(__name__)

HYBRID_SEMANTIC_ONLY_MIN = 0.92
HYBRID_WEAK_JACCARD_MIN = 0.20
HYBRID_STATEMENT_RATIO_MIN = 0.35
DEFAULT_SEMANTIC_UNIT_TYPES = ("function", "method")
SEMANTIC_UNIT_TYPE_TO_ENUM: dict[str, CodeUnitType] = {
    "function": CodeUnitType.FUNCTION,
    "method": CodeUnitType.METHOD,
    "class": CodeUnitType.CLASS,
}
DEFAULT_TINY_UNIT_STATEMENT_CUTOFF = 3
DEFAULT_TINY_NEAR_JACCARD_MIN = 0.93


def _build_exact_hash_exclusions(units: list[CodeUnit]) -> set[tuple[str, str]]:
    """Build exclusion pairs for exact-duplicate units using precomputed hashes.

    :param units: Code units to scan for shared hash pairs.
    :return: Set of unit uid pairs that should be treated as exact duplicates.
    """
    buckets: dict[tuple[str, str], list[CodeUnit]] = {}
    for unit in units:
        ast_hash = unit._ast_hash
        token_hash = unit._token_hash
        if not ast_hash or not token_hash:
            continue
        key = (ast_hash, token_hash)
        buckets.setdefault(key, []).append(unit)

    exclude: set[tuple[str, str]] = set()
    for bucket_units in buckets.values():
        if len(bucket_units) < 2:
            continue
        for i, unit_a in enumerate(bucket_units):
            for unit_b in bucket_units[i + 1 :]:
                exclude.add(ordered_pair_key(unit_a, unit_b))

    return exclude


def _is_test_function_unit(unit: CodeUnit) -> bool:
    """Return whether the unit is a pytest-style test function.

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
    return {
        SEMANTIC_UNIT_TYPE_TO_ENUM[unit_type_name]
        for unit_type_name in semantic_unit_types
        if unit_type_name in SEMANTIC_UNIT_TYPE_TO_ENUM
    }


def _is_tiny_function_like(
    unit: CodeUnit,
    *,
    statement_cutoff: int,
    statement_cache: dict[str, int],
) -> bool:
    """Return whether a unit is a tiny function/method by statement count.

    :param unit: Unit under inspection.
    :param statement_cutoff: Tiny cutoff (exclusive).
    :param statement_cache: Memoized statement counts by unit uid.
    :return: ``True`` when unit is function/method and count is below cutoff.
    """
    if unit.unit_type not in {CodeUnitType.FUNCTION, CodeUnitType.METHOD}:
        return False

    count = statement_cache.get(unit.uid)
    if count is None:
        count = get_code_unit_statement_count(unit)
        statement_cache[unit.uid] = count
    return count < statement_cutoff


def _filter_tiny_traditional_duplicates(
    exact_duplicates: list[DuplicatePair],
    near_duplicates: list[DuplicatePair],
    *,
    statement_cutoff: int,
    tiny_near_jaccard_min: float,
) -> tuple[list[DuplicatePair], list[DuplicatePair]]:
    """Filter tiny wrapper noise from traditional duplicates.

    :param exact_duplicates: Exact traditional duplicate pairs.
    :param near_duplicates: Near traditional duplicate pairs.
    :param statement_cutoff: Tiny cutoff (exclusive).
    :param tiny_near_jaccard_min: Keep floor for tiny near duplicate Jaccard similarity.
    :return: Filtered exact and near duplicate lists.
    """
    statement_cache: dict[str, int] = {}
    filtered_exact: list[DuplicatePair] = []
    filtered_near: list[DuplicatePair] = []

    for duplicate in exact_duplicates:
        tiny_a = _is_tiny_function_like(
            duplicate.unit_a,
            statement_cutoff=statement_cutoff,
            statement_cache=statement_cache,
        )
        tiny_b = _is_tiny_function_like(
            duplicate.unit_b,
            statement_cutoff=statement_cutoff,
            statement_cache=statement_cache,
        )
        if tiny_a and tiny_b:
            continue
        filtered_exact.append(duplicate)

    for duplicate in near_duplicates:
        tiny_a = _is_tiny_function_like(
            duplicate.unit_a,
            statement_cutoff=statement_cutoff,
            statement_cache=statement_cache,
        )
        tiny_b = _is_tiny_function_like(
            duplicate.unit_b,
            statement_cutoff=statement_cutoff,
            statement_cache=statement_cache,
        )
        if tiny_a and tiny_b and duplicate.similarity < tiny_near_jaccard_min:
            continue
        filtered_near.append(duplicate)

    return filtered_exact, filtered_near


def _synthesize_hybrid_duplicates(
    traditional_duplicates: list[DuplicatePair],
    semantic_duplicates: list[DuplicatePair],
    *,
    semantic_threshold: float,
    jaccard_threshold: float,
) -> tuple[list[HybridDuplicate], int]:
    """Build ranked hybrid duplicates from traditional and semantic outputs.

    :param traditional_duplicates: Traditional duplicate pairs (exact + Jaccard).
    :param semantic_duplicates: Semantic duplicate pairs.
    :param semantic_threshold: Minimum semantic similarity used for hybrid tiering.
    :param jaccard_threshold: Minimum Jaccard similarity used for hybrid tiering.
    :return: Tuple of sorted hybrid duplicates and number filtered pairs.
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

        if has_exact:
            tier = "exact"
            confidence = 1.0
        elif jaccard_sim is not None and jaccard_sim >= jaccard_threshold:
            if semantic_sim is not None and semantic_sim >= semantic_threshold:
                tier = "hybrid_confirmed"
                confidence = (0.5 * semantic_sim) + (0.5 * jaccard_sim)
            else:
                tier = "traditional_near"
                confidence = 0.55 + (0.45 * jaccard_sim)
        elif semantic_sim is not None:
            ids_a = identifier_cache.setdefault(unit_a.uid, extract_identifiers(unit_a.source))
            ids_b = identifier_cache.setdefault(unit_b.uid, extract_identifiers(unit_b.source))
            weak_identifier_jaccard = jaccard_similarity(ids_a, ids_b)
            statement_ratio = _statement_count_ratio(unit_a, unit_b)

            if (
                semantic_sim >= HYBRID_SEMANTIC_ONLY_MIN
                and weak_identifier_jaccard >= HYBRID_WEAK_JACCARD_MIN
                and statement_ratio >= HYBRID_STATEMENT_RATIO_MIN
            ):
                tier = "semantic_high_confidence"
                confidence = 0.45 + (0.55 * semantic_sim)

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

    filtered_raw_count = max(0, len(pair_evidence) - len(hybrid_duplicates))
    return hybrid_duplicates, filtered_raw_count


@dataclass
class AnalyzerConfig:
    """Configuration for the code analyzer."""

    # Extraction
    exclude_patterns: list[str] | None = None
    include_private: bool = True

    # Traditional detection
    jaccard_threshold: float = DEFAULT_TRADITIONAL_THRESHOLD

    # Semantic detection
    semantic_threshold: float | None = None
    model_name: str = DEFAULT_MODEL
    semantic_task: str | None = None
    instruction_prefix: str | None = None
    model_revision: str | None = None
    trust_remote_code: bool | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    min_semantic_lines: int = DEFAULT_MIN_SEMANTIC_LINES
    semantic_unit_types: tuple[str, ...] = DEFAULT_SEMANTIC_UNIT_TYPES
    include_stubs: bool = False
    filter_tiny_traditional: bool = True
    tiny_unit_statement_cutoff: int = DEFAULT_TINY_UNIT_STATEMENT_CUTOFF
    tiny_near_jaccard_min: float = DEFAULT_TINY_NEAR_JACCARD_MIN

    # What to run
    run_traditional: bool = True
    run_semantic: bool = True
    run_unused: bool = True
    strict_unused: bool = False
    suppress_test_semantic_matches: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.jaccard_threshold <= 1.0:
            raise ValueError("jaccard_threshold must be in [0.0, 1.0]")

        if self.semantic_threshold is not None and not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be in [0.0, 1.0]")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if self.min_semantic_lines < 0:
            raise ValueError("min_semantic_lines must be >= 0")

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

        if not 0.0 <= self.tiny_near_jaccard_min <= 1.0:
            raise ValueError("tiny_near_jaccard_min must be in [0.0, 1.0]")


class CodeAnalyzer:
    """
    Main analyzer for detecting duplicate and unused code.

    Combines traditional AST-based methods with semantic embedding similarity.
    """

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        """Initialize analyzer state.

        :param config: Optional analyzer configuration override.
        """
        self.config = config or AnalyzerConfig()
        self._units: list[CodeUnit] | None = None
        self._embeddings: np.ndarray | None = None
        self._semantic_units: list[CodeUnit] | None = None
        self._resolved_semantic_threshold: float | None = None
        self._resolved_semantic_task: str | None = None
        self._resolved_search_semantic_task: str | None = None

    def analyze(self, path: Path | str) -> AnalysisResult:
        """
        Run full analysis on a directory or file.

        Args:
            path: Path to directory or single Python file

        Returns:
            AnalysisResult with all findings
        """
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        logger.info(f"Extracting code units from {path}")

        if path.is_file():
            extractor = CodeExtractor(
                path.parent,
                exclude_patterns=self.config.exclude_patterns,
                include_private=self.config.include_private,
            )
            units = list(extractor.extract_from_file(path))
        else:
            extractor = CodeExtractor(
                path,
                exclude_patterns=self.config.exclude_patterns,
                include_private=self.config.include_private,
                include_stubs=self.config.include_stubs,
            )
            units = extractor.extract_all()

        self._units = units
        logger.info(f"Extracted {len(units)} code units")

        if not units:
            return AnalysisResult(
                units=[],
                traditional_duplicates=[],
                semantic_duplicates=[],
                hybrid_duplicates=[],
                potentially_unused=[],
                analysis_mode="none",
                filtered_raw_duplicates=0,
            )

        traditional_duplicates: list[DuplicatePair] = []
        unused: list[CodeUnit] = []
        semantic_threshold = (
            self.config.semantic_threshold
            if self.config.semantic_threshold is not None
            else get_default_semantic_threshold(self.config.model_name)
        )
        semantic_task = self.config.semantic_task or DEFAULT_CHECK_SEMANTIC_TASK
        search_semantic_task = self.config.semantic_task or DEFAULT_SEARCH_SEMANTIC_TASK
        self._resolved_semantic_threshold = semantic_threshold
        self._resolved_semantic_task = semantic_task
        self._resolved_search_semantic_task = search_semantic_task

        semantic_candidates: list[CodeUnit] = []
        self._semantic_units = None
        if self.config.run_semantic:
            semantic_type_filter = _resolve_semantic_unit_type_filter(
                self.config.semantic_unit_types
            )
            semantic_candidates = [
                unit
                for unit in units
                if unit.unit_type in semantic_type_filter
                and get_code_unit_statement_count(unit) >= self.config.min_semantic_lines
            ]
            self._semantic_units = semantic_candidates

        if self.config.run_traditional:
            traditional_duplicate_units = units
            compute_unused_with_traditional = self.config.run_unused
            if self.config.run_semantic:
                # In combined mode, keep traditional duplicate scope aligned with semantic scope.
                traditional_duplicate_units = semantic_candidates
                # Unused analysis should still operate on the full extraction set.
                compute_unused_with_traditional = False

            exact_dupes, near_dupes, unused = run_traditional_analysis(
                traditional_duplicate_units,
                jaccard_threshold=self.config.jaccard_threshold,
                compute_unused=compute_unused_with_traditional,
                project_root=path,
                strict_unused=self.config.strict_unused,
            )
            if self.config.filter_tiny_traditional:
                exact_dupes, near_dupes = _filter_tiny_traditional_duplicates(
                    exact_dupes,
                    near_dupes,
                    statement_cutoff=self.config.tiny_unit_statement_cutoff,
                    tiny_near_jaccard_min=self.config.tiny_near_jaccard_min,
                )
            traditional_duplicates = exact_dupes + near_dupes

            if self.config.run_semantic and self.config.run_unused:
                build_reference_graph(units, project_root=path)
                unused = find_potentially_unused(units, strict_unused=self.config.strict_unused)
        elif self.config.run_unused:
            build_reference_graph(units, project_root=path)
            unused = find_potentially_unused(units, strict_unused=self.config.strict_unused)

        semantic_duplicates: list[DuplicatePair] = []

        if self.config.run_semantic:
            exclude: set[tuple[str, str]] = set()

            if self.config.run_traditional:
                exclude = {
                    ordered_pair_key(duplicate.unit_a, duplicate.unit_b)
                    for duplicate in traditional_duplicates
                    if duplicate.method in {"ast_hash", "token_hash"}
                }
                # Keep near-duplicate pairs out of exclusion so semantic scoring can confirm
                # traditional evidence and enable hybrid_confirmed scoring.
                exclude.update(_build_exact_hash_exclusions(semantic_candidates))

            try:
                semantic_kwargs: dict[str, object] = {
                    "model_name": self.config.model_name,
                    "instruction_prefix": self.config.instruction_prefix,
                    "threshold": semantic_threshold,
                    "exclude_pairs": exclude,
                    "batch_size": self.config.batch_size,
                    "revision": self.config.model_revision,
                    "trust_remote_code": self.config.trust_remote_code,
                }
                if self.config.semantic_task is not None:
                    semantic_kwargs["semantic_task"] = semantic_task
                self._embeddings, semantic_duplicates = run_semantic_analysis(
                    semantic_candidates,
                    **semantic_kwargs,
                )
            except (ModuleNotFoundError, SemanticBackendError, RuntimeError) as exc:
                # If semantic is the only duplicate-detection method requested,
                # fail hard instead of silently degrading to unused-only output.
                if not self.config.run_traditional:
                    raise

                self._embeddings = None
                semantic_duplicates = []
                runtime_versions = get_semantic_runtime_versions()
                version_text = ", ".join(
                    f"{key}={value}" for key, value in runtime_versions.items()
                )
                logger.warning(
                    "Semantic analysis unavailable (%s). Proceeding with non-semantic analysis. "
                    "model=%s revision=%s trust_remote_code=%s [%s]. "
                    "Retry with `codedupes check %s --traditional-only`.",
                    exc,
                    self.config.model_name,
                    self.config.model_revision,
                    self.config.trust_remote_code,
                    version_text,
                    path,
                )

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
        filtered_raw_duplicates = 0

        if combined_mode:
            hybrid_duplicates, filtered_raw_duplicates = _synthesize_hybrid_duplicates(
                traditional_duplicates,
                semantic_duplicates,
                semantic_threshold=semantic_threshold,
                jaccard_threshold=self.config.jaccard_threshold,
            )

        if combined_mode:
            analysis_mode = "combined"
        elif self.config.run_traditional:
            analysis_mode = "traditional"
        elif self.config.run_semantic:
            analysis_mode = "semantic"
        else:
            analysis_mode = "none"

        return AnalysisResult(
            units=units,
            traditional_duplicates=traditional_duplicates,
            semantic_duplicates=semantic_duplicates,
            hybrid_duplicates=hybrid_duplicates,
            potentially_unused=unused,
            analysis_mode=analysis_mode,
            filtered_raw_duplicates=filtered_raw_duplicates,
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[CodeUnit, float]]:
        """
        Search for code units matching a natural language query.

        Must run analyze() first to compute embeddings.

        :param query: Search query string.
        :param top_k: Maximum results to return.
        :return: List of code units and cosine scores.
        """
        if self._units is None or self._embeddings is None:
            raise RuntimeError("Must run analyze() with run_semantic=True before search().")

        if not self._semantic_units:
            return []

        from codedupes.semantic import find_similar_to_query

        if self._resolved_semantic_threshold is None or self._resolved_search_semantic_task is None:
            raise RuntimeError("Semantic configuration was not resolved; run analyze() first.")

        return find_similar_to_query(
            query,
            self._semantic_units,
            self._embeddings,
            model_name=self.config.model_name,
            instruction_prefix=self.config.instruction_prefix,
            top_k=top_k,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
            threshold=self._resolved_semantic_threshold,
            semantic_task=self._resolved_search_semantic_task,
        )


def analyze_directory(
    path: Path | str,
    semantic_threshold: float | None = None,
    traditional_threshold: float = DEFAULT_TRADITIONAL_THRESHOLD,
    exclude_patterns: list[str] | None = None,
    model_name: str = DEFAULT_MODEL,
    semantic_task: str | None = None,
    instruction_prefix: str | None = None,
    model_revision: str | None = None,
    trust_remote_code: bool | None = None,
    min_semantic_lines: int = DEFAULT_MIN_SEMANTIC_LINES,
    semantic_unit_types: tuple[str, ...] = DEFAULT_SEMANTIC_UNIT_TYPES,
    filter_tiny_traditional: bool = True,
    tiny_unit_statement_cutoff: int = DEFAULT_TINY_UNIT_STATEMENT_CUTOFF,
    tiny_near_jaccard_min: float = DEFAULT_TINY_NEAR_JACCARD_MIN,
    include_stubs: bool = False,
    run_unused: bool = True,
    strict_unused: bool = False,
) -> AnalysisResult:
    """
    Convenience function for quick analysis.

    Args:
        path: Directory to analyze
        semantic_threshold: Cosine similarity threshold for semantic duplicates
        traditional_threshold: Jaccard threshold for traditional near-duplicates
        exclude_patterns: Glob patterns for files to exclude
        model_name: HuggingFace model for embeddings
        semantic_task: Semantic task mode for prompt/inference behavior
        instruction_prefix: Custom instruction prefix prepended to semantic inputs
        model_revision: Optional HuggingFace model revision/commit hash.
            If None, semantic backend chooses model-specific default behavior.
        trust_remote_code: Whether remote model code may execute while loading
        min_semantic_lines: Minimum statement count required for semantic analysis.
        semantic_unit_types: Unit types eligible for semantic embeddings.
        filter_tiny_traditional: Filter tiny traditional duplicates when true.
        tiny_unit_statement_cutoff: Tiny function/method cutoff (exclusive).
        tiny_near_jaccard_min: Keep floor for tiny near-duplicate Jaccard pairs.
        include_stubs: Whether to analyze ``.pyi`` files.
        strict_unused: Whether to ignore public API exclusions when reporting unused code.
        run_unused: Run potentially-unused detection even when traditional analysis is off

    Returns:
        AnalysisResult
    """
    config = AnalyzerConfig(
        semantic_threshold=semantic_threshold,
        jaccard_threshold=traditional_threshold,
        exclude_patterns=exclude_patterns,
        model_name=model_name,
        semantic_task=semantic_task,
        instruction_prefix=instruction_prefix,
        model_revision=model_revision,
        trust_remote_code=trust_remote_code,
        min_semantic_lines=min_semantic_lines,
        semantic_unit_types=semantic_unit_types,
        filter_tiny_traditional=filter_tiny_traditional,
        tiny_unit_statement_cutoff=tiny_unit_statement_cutoff,
        tiny_near_jaccard_min=tiny_near_jaccard_min,
        include_stubs=include_stubs,
        run_unused=run_unused,
        strict_unused=strict_unused,
    )

    analyzer = CodeAnalyzer(config)
    return analyzer.analyze(path)
