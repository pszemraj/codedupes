"""Data models for extracted code units and analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from codedupes.pairs import unordered_pair_key

if TYPE_CHECKING:
    from codedupes.semantic import EmbeddingRunStats


class CodeUnitType(Enum):
    """Kinds of analyzed code units."""

    FUNCTION = auto()
    METHOD = auto()
    CLASS = auto()


DiagnosticSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class ExtractionDiagnostic:
    """A recoverable or fatal issue observed while processing one source file.

    Extraction reports parse problems here; later stages reuse the same shape for
    per-unit warnings, such as backend truncation of a definition that remains
    eligible for semantic comparison.
    """

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

    Every language backend produces the same shape: the exact node span of the
    definition as its grammar delimits it (a Python decorated definition starts
    at its first decorator) plus the features computed while the syntax tree
    was in hand, so downstream duplicate and semantic stages never reparse
    source in a language-specific way.
    """

    name: str
    qualified_name: str
    unit_type: CodeUnitType
    file_path: Path
    lineno: int
    end_lineno: int
    source: str

    # Language and source-range metadata. The defaults let tests and callers
    # build a unit by hand without spelling out every backend field.
    language: str = "python"
    dialect: str | None = None
    native_kind: str | None = None
    start_byte: int = 0
    end_byte: int = 0
    start_column: int = 0
    end_column: int = 0
    statement_count: int | None = None

    # Fingerprints from the shared canonical stream: the structural hash
    # normalizes local names and strips formatting and docstrings, while the
    # token hash keeps every token that survives the language's hash policy.
    structural_hash: str | None = field(default=None, repr=False)
    token_hash: str | None = field(default=None, repr=False)
    identifiers: frozenset[str] = field(default_factory=frozenset, repr=False)

    # Populated by unused-code analysis, which is Python-only by design.
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

        :return: Identifier that is unique to this unit within one analysis run.
        """
        return f"{self.file_path}::{self.language}::{self.qualified_name}::{self.start_byte}"

    @property
    def is_likely_api(self) -> bool:
        """Indicate whether this unit is likely public API surface.

        :return: ``True`` when the unit looks externally reachable.
        """
        return (
            self.is_exported
            or self.is_dunder
            or (self.is_public and self.unit_type == CodeUnitType.CLASS)
            or self.name in ("__init__", "__new__", "__call__")
        )

    def overlaps(self, other: CodeUnit) -> bool:
        """Return whether two units occupy overlapping source ranges.

        :param other: Unit to compare against.
        :return: ``True`` when both units share source range in the same file.
        """
        if self.file_path != other.file_path:
            return False
        if self.end_byte > self.start_byte and other.end_byte > other.start_byte:
            return self.start_byte < other.end_byte and other.start_byte < self.end_byte
        return self.lineno <= other.end_lineno and other.lineno <= self.end_lineno


class _PairIdentity:
    """Hash and compare a duplicate record by its unordered unit pair.

    Score payloads are deliberately excluded from identity so re-scored
    records of the same pair dedupe in sets and dict keys.
    """

    unit_a: CodeUnit
    unit_b: CodeUnit

    def __hash__(self) -> int:
        return hash(unordered_pair_key(self.unit_a, self.unit_b))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return unordered_pair_key(self.unit_a, self.unit_b) == unordered_pair_key(
            other.unit_a, other.unit_b
        )


@dataclass(eq=False)
class DuplicatePair(_PairIdentity):
    """A pair of code units identified as duplicates."""

    unit_a: CodeUnit
    unit_b: CodeUnit
    similarity: float
    method: str


HybridTier = Literal[
    "exact",
    "traditional_near",
    "hybrid_confirmed",
    "semantic_high_confidence",
    "semantic_review",
]
# Runtime view of ``HybridTier`` in declaration order, for zero-filled tier counts.
HYBRID_TIERS: tuple[HybridTier, ...] = get_args(HybridTier)

AnalysisMode = Literal["combined", "traditional", "semantic", "none"]


@dataclass(eq=False)
class HybridDuplicate(_PairIdentity):
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


@dataclass
class AnalysisResult:
    """Full analysis result."""

    units: list[CodeUnit]
    traditional_duplicates: list[DuplicatePair]
    semantic_duplicates: list[DuplicatePair]
    hybrid_duplicates: list[HybridDuplicate]
    potentially_unused: list[CodeUnit]
    analysis_mode: AnalysisMode
    semantic_fallback: bool = False
    semantic_fallback_reason: str | None = None
    extraction_diagnostics: list[ExtractionDiagnostic] = field(default_factory=list)
    semantic_diagnostics: list[ExtractionDiagnostic] = field(default_factory=list)
    unused_supported_languages: tuple[str, ...] = ("python",)
    unused_excluded_units: int = 0
    embedding_stats: EmbeddingRunStats | None = None

    @property
    def all_duplicates(self) -> list[HybridDuplicate] | list[DuplicatePair]:
        """Return the available duplicate list for this analysis mode.

        :return: Hybrid duplicates in combined mode, otherwise traditional plus semantic pairs.
        """
        if self.analysis_mode == "combined":
            return self.hybrid_duplicates
        return self.traditional_duplicates + self.semantic_duplicates
