"""Main analyzer orchestrating all detection methods."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codedupes.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_C2LLM_REVISION,
    DEFAULT_MIN_SEMANTIC_LINES,
    DEFAULT_MODEL,
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_TRADITIONAL_THRESHOLD,
)
from codedupes.extractor import CodeExtractor
from codedupes.models import AnalysisResult, CodeUnit, CodeUnitType, DuplicatePair, HybridDuplicate
from codedupes.semantic import (
    SemanticBackendError,
    get_code_unit_statement_count,
    get_semantic_runtime_versions,
    run_semantic_analysis,
)
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


def _pair_key(unit_a: CodeUnit, unit_b: CodeUnit) -> tuple[str, str]:
    """Return stable key for an unordered pair."""
    return (min(unit_a.uid, unit_b.uid), max(unit_a.uid, unit_b.uid))


def _build_exact_hash_exclusions(units: list[CodeUnit]) -> set[tuple[str, str]]:
    """Build exclusion pairs for exact-duplicate units using precomputed hashes."""
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
                exclude.add(_pair_key(unit_a, unit_b))

    return exclude


def _is_test_function_unit(unit: CodeUnit) -> bool:
    """Return True when the unit looks like a pytest-style test function."""
    return unit.unit_type in {CodeUnitType.FUNCTION, CodeUnitType.METHOD} and unit.name.startswith(
        "test_"
    )


def _statement_count_ratio(unit_a: CodeUnit, unit_b: CodeUnit) -> float:
    """Compute ratio of statement counts for two units."""
    count_a = get_code_unit_statement_count(unit_a)
    count_b = get_code_unit_statement_count(unit_b)
    high = max(count_a, count_b)
    low = min(count_a, count_b)
    if high == 0:
        return 0.0
    return low / high


def _synthesize_hybrid_duplicates(
    traditional_duplicates: list[DuplicatePair],
    semantic_duplicates: list[DuplicatePair],
    *,
    semantic_threshold: float,
    jaccard_threshold: float,
) -> tuple[list[HybridDuplicate], int]:
    """Build a single ranked hybrid duplicate list from raw method outputs."""
    pair_evidence: dict[tuple[str, str], dict[str, object]] = {}

    def ensure_entry(unit_a: CodeUnit, unit_b: CodeUnit) -> dict[str, object]:
        key = _pair_key(unit_a, unit_b)
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
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD
    model_name: str = DEFAULT_MODEL
    instruction_prefix: str | None = None
    model_revision: str | None = DEFAULT_C2LLM_REVISION
    trust_remote_code: bool | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    min_semantic_lines: int = DEFAULT_MIN_SEMANTIC_LINES
    include_stubs: bool = False

    # What to run
    run_traditional: bool = True
    run_semantic: bool = True
    run_unused: bool = True
    strict_unused: bool = False
    suppress_test_semantic_matches: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.jaccard_threshold <= 1.0:
            raise ValueError("jaccard_threshold must be in [0.0, 1.0]")

        if not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be in [0.0, 1.0]")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if self.min_semantic_lines < 0:
            raise ValueError("min_semantic_lines must be >= 0")


class CodeAnalyzer:
    """
    Main analyzer for detecting duplicate and unused code.

    Combines traditional AST-based methods with semantic embedding similarity.
    """

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        self.config = config or AnalyzerConfig()
        self._units: list[CodeUnit] | None = None
        self._embeddings: np.ndarray | None = None
        self._semantic_units: list[CodeUnit] | None = None

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
                filtered_raw_duplicates=0,
            )

        traditional_duplicates: list[DuplicatePair] = []
        unused: list[CodeUnit] = []

        if self.config.run_traditional:
            exact_dupes, near_dupes, unused = run_traditional_analysis(
                units,
                jaccard_threshold=self.config.jaccard_threshold,
                compute_unused=self.config.run_unused,
                project_root=path,
                strict_unused=self.config.strict_unused,
            )
            traditional_duplicates = exact_dupes + near_dupes
        elif self.config.run_unused:
            build_reference_graph(units, project_root=path)
            unused = find_potentially_unused(units, strict_unused=self.config.strict_unused)

        semantic_duplicates: list[DuplicatePair] = []
        self._semantic_units = None

        if self.config.run_semantic:
            semantic_candidates = [
                unit
                for unit in units
                if get_code_unit_statement_count(unit) >= self.config.min_semantic_lines
            ]
            self._semantic_units = semantic_candidates

            exclude: set[tuple[str, str]] = {
                _pair_key(duplicate.unit_a, duplicate.unit_b)
                for duplicate in traditional_duplicates
            }
            exclude.update(_build_exact_hash_exclusions(semantic_candidates))

            try:
                self._embeddings, semantic_duplicates = run_semantic_analysis(
                    semantic_candidates,
                    model_name=self.config.model_name,
                    instruction_prefix=self.config.instruction_prefix,
                    threshold=self.config.semantic_threshold,
                    exclude_pairs=exclude,
                    batch_size=self.config.batch_size,
                    revision=self.config.model_revision,
                    trust_remote_code=self.config.trust_remote_code,
                )
            except (ModuleNotFoundError, SemanticBackendError) as exc:
                if not self.config.run_traditional and not self.config.run_unused:
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
                    "Retry with `codedupes check %s --traditional-only` or install compatible "
                    "deps: pip install 'transformers>=4.51,<5' 'sentence-transformers>=5,<6'.",
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
                semantic_threshold=self.config.semantic_threshold,
                jaccard_threshold=self.config.jaccard_threshold,
            )

        return AnalysisResult(
            units=units,
            traditional_duplicates=traditional_duplicates,
            semantic_duplicates=semantic_duplicates,
            hybrid_duplicates=hybrid_duplicates,
            potentially_unused=unused,
            filtered_raw_duplicates=filtered_raw_duplicates,
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[CodeUnit, float]]:
        """
        Search for code units matching a natural language query.

        Must run analyze() first to compute embeddings.
        """
        if self._units is None or self._embeddings is None:
            raise RuntimeError("Must run analyze() with run_semantic=True before search().")

        if not self._semantic_units:
            return []

        from codedupes.semantic import find_similar_to_query

        return find_similar_to_query(
            query,
            self._semantic_units,
            self._embeddings,
            model_name=self.config.model_name,
            instruction_prefix=self.config.instruction_prefix,
            top_k=top_k,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
        )


def analyze_directory(
    path: Path | str,
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    traditional_threshold: float = DEFAULT_TRADITIONAL_THRESHOLD,
    exclude_patterns: list[str] | None = None,
    model_name: str = DEFAULT_MODEL,
    instruction_prefix: str | None = None,
    model_revision: str | None = DEFAULT_C2LLM_REVISION,
    trust_remote_code: bool | None = None,
    min_semantic_lines: int = DEFAULT_MIN_SEMANTIC_LINES,
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
        instruction_prefix: Custom instruction prefix prepended to semantic inputs
        model_revision: HuggingFace model revision/commit hash
        trust_remote_code: Whether remote model code may execute while loading
        run_unused: Run potentially-unused detection even when traditional analysis is off

    Returns:
        AnalysisResult
    """
    config = AnalyzerConfig(
        semantic_threshold=semantic_threshold,
        jaccard_threshold=traditional_threshold,
        exclude_patterns=exclude_patterns,
        model_name=model_name,
        instruction_prefix=instruction_prefix,
        model_revision=model_revision,
        trust_remote_code=trust_remote_code,
        min_semantic_lines=min_semantic_lines,
        include_stubs=include_stubs,
        run_unused=run_unused,
        strict_unused=strict_unused,
    )

    analyzer = CodeAnalyzer(config)
    return analyzer.analyze(path)
