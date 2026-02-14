"""Main analyzer orchestrating all detection methods."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codedupes.extractor import CodeExtractor
from codedupes.models import AnalysisResult, CodeUnit, DuplicatePair
from codedupes.semantic import get_code_unit_statement_count, run_semantic_analysis
from codedupes.traditional import (
    build_reference_graph,
    find_potentially_unused,
    run_traditional_analysis,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "codefuse-ai/C2LLM-0.5B"


@dataclass
class AnalyzerConfig:
    """Configuration for the code analyzer."""

    # Extraction
    exclude_patterns: list[str] | None = None
    include_private: bool = True

    # Traditional detection
    jaccard_threshold: float = 0.85

    # Semantic detection
    semantic_threshold: float = 0.82
    model_name: str = DEFAULT_MODEL
    batch_size: int = 32
    min_semantic_lines: int = 3
    include_stubs: bool = False

    # What to run
    run_traditional: bool = True
    run_semantic: bool = True
    run_unused: bool = True
    strict_unused: bool = False

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

        # Extract code units
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
                exact_duplicates=[],
                semantic_duplicates=[],
                potentially_unused=[],
            )

        # Run traditional analysis
        exact_dupes: list[DuplicatePair] = []
        near_dupes: list[DuplicatePair] = []
        unused: list[CodeUnit] = []

        if self.config.run_traditional:
            exact_dupes, near_dupes, unused = run_traditional_analysis(
                units,
                jaccard_threshold=self.config.jaccard_threshold,
                compute_unused=self.config.run_unused,
                project_root=path,
                strict_unused=self.config.strict_unused,
            )
        elif self.config.run_unused:
            build_reference_graph(units, project_root=path)
            unused = find_potentially_unused(units, strict_unused=self.config.strict_unused)

        # Run semantic analysis
        semantic_dupes: list[DuplicatePair] = []
        self._semantic_units = None

        if self.config.run_semantic:
            semantic_candidates = [
                unit
                for unit in units
                if get_code_unit_statement_count(unit) >= self.config.min_semantic_lines
            ]
            self._semantic_units = semantic_candidates

            # Exclude pairs already found by exact methods
            exclude: set[tuple[str, str]] = {
                (min(d.unit_a.uid, d.unit_b.uid), max(d.unit_a.uid, d.unit_b.uid))
                for d in exact_dupes
            }
            exclude.update(
                {
                    (min(d.unit_a.uid, d.unit_b.uid), max(d.unit_a.uid, d.unit_b.uid))
                    for d in near_dupes
                }
            )

            try:
                self._embeddings, semantic_dupes = run_semantic_analysis(
                    semantic_candidates,
                    model_name=self.config.model_name,
                    threshold=self.config.semantic_threshold,
                    exclude_pairs=exclude,
                    batch_size=self.config.batch_size,
                )
            except ModuleNotFoundError:
                # Semantic analysis is best-effort unless it is the only requested mode.
                if not self.config.run_traditional and not self.config.run_unused:
                    raise

                self._embeddings = None
                semantic_dupes = []
                logger.warning(
                    "Semantic dependencies unavailable; proceeding with non-semantic analysis only."
                )

            if self.config.run_unused:
                unused_uids = {unit.uid for unit in unused}
                semantic_dupes = [
                    duplicate
                    for duplicate in semantic_dupes
                    if not (
                        duplicate.unit_a.uid in unused_uids and duplicate.unit_b.uid in unused_uids
                    )
                ]

        # Combine exact + near for "exact" category
        all_exact = exact_dupes + near_dupes

        return AnalysisResult(
            units=units,
            exact_duplicates=all_exact,
            semantic_duplicates=semantic_dupes,
            potentially_unused=unused,
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
            top_k=top_k,
        )


def analyze_directory(
    path: Path | str,
    semantic_threshold: float = 0.82,
    traditional_threshold: float = 0.85,
    exclude_patterns: list[str] | None = None,
    model_name: str = DEFAULT_MODEL,
    min_semantic_lines: int = 3,
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
        run_unused: Run potentially-unused detection even when traditional analysis is off

    Returns:
        AnalysisResult
    """
    config = AnalyzerConfig(
        semantic_threshold=semantic_threshold,
        jaccard_threshold=traditional_threshold,
        exclude_patterns=exclude_patterns,
        model_name=model_name,
        min_semantic_lines=min_semantic_lines,
        include_stubs=include_stubs,
        run_unused=run_unused,
        strict_unused=strict_unused,
    )

    analyzer = CodeAnalyzer(config)
    return analyzer.analyze(path)
