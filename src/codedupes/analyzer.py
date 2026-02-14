"""Main analyzer orchestrating all detection methods."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .extractor import CodeExtractor
from .models import AnalysisResult, CodeUnit, DuplicatePair
from .semantic import run_semantic_analysis
from .traditional import run_traditional_analysis

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
    semantic_threshold: float = 0.85
    model_name: str = DEFAULT_MODEL
    batch_size: int = 32

    # What to run
    run_traditional: bool = True
    run_semantic: bool = True


class CodeAnalyzer:
    """
    Main analyzer for detecting duplicate and unused code.

    Combines traditional AST-based methods with semantic embedding similarity.
    """

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        self.config = config or AnalyzerConfig()
        self._units: list[CodeUnit] | None = None
        self._embeddings: np.ndarray | None = None

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
            )

        # Run semantic analysis
        semantic_dupes: list[DuplicatePair] = []

        if self.config.run_semantic:
            # Exclude pairs already found by exact methods
            exclude: set[tuple[str, str]] = {
                (min(d.unit_a.uid, d.unit_b.uid), max(d.unit_a.uid, d.unit_b.uid))
                for d in exact_dupes
            }

            self._embeddings, semantic_dupes = run_semantic_analysis(
                units,
                model_name=self.config.model_name,
                threshold=self.config.semantic_threshold,
                exclude_pairs=exclude,
                batch_size=self.config.batch_size,
            )

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
            raise RuntimeError("Must run analyze() before search()")

        from .semantic import find_similar_to_query

        return find_similar_to_query(
            query,
            self._units,
            self._embeddings,
            model_name=self.config.model_name,
            top_k=top_k,
        )


def analyze_directory(
    path: Path | str,
    semantic_threshold: float = 0.85,
    traditional_threshold: float = 0.85,
    exclude_patterns: list[str] | None = None,
    model_name: str = DEFAULT_MODEL,
) -> AnalysisResult:
    """
    Convenience function for quick analysis.

    Args:
        path: Directory to analyze
        semantic_threshold: Cosine similarity threshold for semantic duplicates
        traditional_threshold: Jaccard threshold for traditional near-duplicates
        exclude_patterns: Glob patterns for files to exclude
        model_name: HuggingFace model for embeddings

    Returns:
        AnalysisResult
    """
    config = AnalyzerConfig(
        semantic_threshold=semantic_threshold,
        jaccard_threshold=traditional_threshold,
        exclude_patterns=exclude_patterns,
        model_name=model_name,
    )

    analyzer = CodeAnalyzer(config)
    return analyzer.analyze(path)
