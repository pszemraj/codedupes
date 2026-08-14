"""Data models for extracted code units and analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Literal

from codedupes.pairs import unordered_pair_key


class CodeUnitType(Enum):
    """Kinds of analyzed code units."""

    FUNCTION = auto()
    METHOD = auto()
    CLASS = auto()


DiagnosticSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class ExtractionDiagnostic:
    """A recoverable or fatal issue observed while extracting one source file."""

    file_path: Path
    language: str
    message: str
    severity: DiagnosticSeverity = "warning"
    code: str = "parse-warning"
    lineno: int | None = None
    end_lineno: int | None = None


@dataclass
class CodeUnit:
    """Represents an extracted function, method, or class.

    Python remains the compatibility baseline, while the language-neutral fields
    make the same model usable by Tree-sitter backends.  Backend-computed
    features are stored on the unit so downstream duplicate and semantic stages
    never need to reparse source in a language-specific way.
    """

    name: str
    qualified_name: str
    unit_type: CodeUnitType
    file_path: Path
    lineno: int
    end_lineno: int
    source: str
    docstring: str | None = None

    # Language and source-range metadata. Defaults preserve source compatibility
    # for callers that manually construct Python CodeUnit instances.
    language: str = "python"
    dialect: str | None = None
    native_kind: str | None = None
    start_byte: int = 0
    end_byte: int = 0
    start_column: int = 0
    end_column: int = 0
    has_body: bool = True
    statement_count: int | None = None

    # Backend-computed structural/token fingerprints. Python derives both from
    # its normalized CPython AST; Tree-sitter backends use the canonical
    # fingerprint stream.
    structural_hash: str | None = field(default=None, repr=False)
    token_hash: str | None = field(default=None, repr=False)
    identifiers: frozenset[str] = field(default_factory=frozenset, repr=False)

    # For call graph / usage analysis.  Reference resolution is intentionally
    # Python-only in the first polyglot release.
    calls: set[str] = field(default_factory=set)
    references: set[str] = field(default_factory=set)

    # API exposure markers
    is_public: bool = False
    is_dunder: bool = False
    is_exported: bool = False

    @property
    def uid(self) -> str:
        """Build an in-run unique identifier for this code unit.

        The byte position keeps the uid unique for overloads, conditional
        redefinitions, and repeated lexical names, all of which are legal in
        several supported languages (including Python).
        """
        return f"{self.file_path}::{self.language}::{self.qualified_name}::{self.start_byte}"

    @property
    def is_likely_api(self) -> bool:
        """Indicate whether this unit is likely public API surface."""
        return (
            self.is_exported
            or self.is_dunder
            or (self.is_public and self.unit_type == CodeUnitType.CLASS)
            or self.name in ("__init__", "__new__", "__call__")
        )

    def overlaps(self, other: CodeUnit) -> bool:
        """Return whether two units occupy overlapping source ranges."""
        if self.file_path != other.file_path:
            return False
        if self.end_byte > self.start_byte and other.end_byte > other.start_byte:
            return self.start_byte < other.end_byte and other.start_byte < self.end_byte
        return self.lineno <= other.end_lineno and other.lineno <= self.end_lineno


@dataclass
class DuplicatePair:
    """A pair of code units identified as duplicates."""

    unit_a: CodeUnit
    unit_b: CodeUnit
    similarity: float
    method: str

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
    traditional_duplicates: list[DuplicatePair]
    semantic_duplicates: list[DuplicatePair]
    hybrid_duplicates: list[HybridDuplicate]
    potentially_unused: list[CodeUnit]
    analysis_mode: AnalysisMode
    filtered_raw_duplicates: int = 0
    semantic_fallback: bool = False
    semantic_fallback_reason: str | None = None
    extraction_diagnostics: list[ExtractionDiagnostic] = field(default_factory=list)
    unused_supported_languages: tuple[str, ...] = ("python",)
    unused_excluded_units: int = 0

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
