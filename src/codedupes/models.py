"""Data models for extracted code units."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Literal

from codedupes.pairs import unordered_pair_key


class CodeUnitType(Enum):
    FUNCTION = auto()
    METHOD = auto()
    CLASS = auto()


@dataclass
class CodeUnit:
    """Represents an extracted function, method, or class."""

    name: str
    qualified_name: str  # module.ClassName.method_name
    unit_type: CodeUnitType
    file_path: Path
    lineno: int
    end_lineno: int
    source: str
    docstring: str | None = None

    # Computed on demand
    _ast_hash: str | None = field(default=None, repr=False)
    _token_hash: str | None = field(default=None, repr=False)

    # For call graph / usage analysis
    calls: set[str] = field(default_factory=set)
    references: set[str] = field(default_factory=set)  # who calls this

    # API exposure markers
    is_public: bool = False
    is_dunder: bool = False
    is_exported: bool = False  # in __all__

    @property
    def uid(self) -> str:
        """Unique identifier for this code unit."""
        return f"{self.file_path}::{self.qualified_name}"

    @property
    def is_likely_api(self) -> bool:
        """Heuristic: is this likely intentionally exposed?"""
        return (
            self.is_exported
            or self.is_dunder
            or (self.is_public and self.unit_type == CodeUnitType.CLASS)
            or self.name in ("__init__", "__new__", "__call__")
        )


@dataclass
class DuplicatePair:
    """A pair of code units identified as duplicates."""

    unit_a: CodeUnit
    unit_b: CodeUnit
    similarity: float
    method: str  # "ast_hash", "token_hash", "semantic"

    def __hash__(self) -> int:
        return hash(unordered_pair_key(self.unit_a, self.unit_b))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DuplicatePair):
            return False
        return unordered_pair_key(self.unit_a, self.unit_b) == unordered_pair_key(
            other.unit_a, other.unit_b
        )


HybridTier = Literal[
    "exact",
    "traditional_near",
    "hybrid_confirmed",
    "semantic_high_confidence",
]

AnalysisMode = Literal["combined", "traditional", "semantic", "none"]


@dataclass
class HybridDuplicate:
    """A synthesized duplicate candidate combining traditional + semantic evidence."""

    unit_a: CodeUnit
    unit_b: CodeUnit
    tier: HybridTier
    confidence: float
    has_exact: bool = False
    jaccard_similarity: float | None = None
    semantic_similarity: float | None = None
    weak_identifier_jaccard: float | None = None
    statement_count_ratio: float | None = None

    def __hash__(self) -> int:
        return hash(unordered_pair_key(self.unit_a, self.unit_b))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HybridDuplicate):
            return False
        return unordered_pair_key(self.unit_a, self.unit_b) == unordered_pair_key(
            other.unit_a, other.unit_b
        )


@dataclass
class AnalysisResult:
    """Full analysis result."""

    units: list[CodeUnit]
    traditional_duplicates: list[DuplicatePair]  # AST/token/jaccard matches
    semantic_duplicates: list[DuplicatePair]  # Embedding similarity
    hybrid_duplicates: list[HybridDuplicate]  # Final combined output candidates
    potentially_unused: list[CodeUnit]  # No references, not API
    analysis_mode: AnalysisMode  # How duplicates were synthesized
    filtered_raw_duplicates: int = 0

    @property
    def exact_duplicates(self) -> list[DuplicatePair]:
        """Backward-compatible alias for traditional duplicates."""
        return self.traditional_duplicates

    @property
    def all_duplicates(self) -> list[HybridDuplicate] | list[DuplicatePair]:
        """Return the available duplicate list for this analysis mode."""
        if self.analysis_mode == "combined":
            return self.hybrid_duplicates
        return self.traditional_duplicates + self.semantic_duplicates
