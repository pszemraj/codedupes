"""Data models for extracted code units and analysis results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from codedupes.pairs import unordered_pair_key

if TYPE_CHECKING:
    from codedupes.semantic import EmbeddingRunStats

# Extraction diagnostic codes that leave a run's scope incomplete rather than
# merely advisory (``c-header-policy``, ``semantic-context-overflow``, and
# ``suppression-syntax`` are notices, not scope loss).
INCOMPLETE_EXTRACTION_CODES: frozenset[str] = frozenset(
    {"read-error", "invalid-utf8", "partial-parse", "unit-parse-error", "walk-error"}
)


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

    # ``codedupes: ignore[...]`` directives attached to this unit, including
    # any inherited from an enclosing unit's own directive.
    suppressions: frozenset[str] = field(default_factory=frozenset)

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

    def source_lines(self, limit: int | None) -> tuple[list[str], int]:
        """Return this unit's source split into lines, bounded by a line budget.

        :param limit: Maximum lines to keep, or ``None`` for no bound.
        :return: Kept lines, and the count of lines omitted from the end.
        """
        lines = self.source.split("\n")
        if limit is None or len(lines) <= limit:
            return lines, 0
        return lines[:limit], len(lines) - limit


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

AnalysisMode = Literal["combined", "traditional", "semantic", "unused"]
CheckStatus = Literal["completed", "partial", "empty", "fallback", "disabled"]
AnalysisStatus = Literal["complete", "partial", "empty"]


@dataclass(eq=False)
class HybridDuplicate(_PairIdentity):
    """A synthesized duplicate candidate combining traditional + semantic evidence."""

    unit_a: CodeUnit
    unit_b: CodeUnit
    tier: HybridTier
    score: float
    # Fingerprint that made the pair exact (``structural_hash`` or ``token_hash``);
    # ``None`` for every other tier.
    exact_method: str | None = None
    jaccard_similarity: float | None = None
    semantic_similarity: float | None = None
    weak_identifier_jaccard: float | None = None
    statement_count_ratio: float | None = None


@dataclass(frozen=True)
class TraditionalSettings:
    """Resolved traditional-detection settings for one run."""

    jaccard_threshold: float
    tiny_filter: bool
    tiny_cutoff: int


@dataclass(frozen=True)
class HybridSplit:
    """Resolved tier-split settings hybrid synthesis applied to semantic-only pairs."""

    weak_identifier_jaccard_min: float
    statement_ratio_min: float
    # Per-language similarity that promotes an uncorroborated pair; a missing
    # language means promotion was off for it.
    promotion_gates: Mapping[str, float]


@dataclass(frozen=True)
class SemanticSettings:
    """Resolved semantic-detection settings for one run."""

    requested_model: str
    model: str
    revision: str | None
    profile: str
    threshold_profile: str
    task: str
    device: str
    execution_device: str | None
    thresholds: Mapping[str, float]
    threshold_floor: float | None
    min_statements: int
    unit_types: tuple[str, ...]
    cross_language: bool
    hybrid_split: HybridSplit | None


@dataclass(frozen=True)
class UnusedSettings:
    """Resolved unused-detection settings for one run."""

    strict: bool
    # Python files parsed for the reference graph, including reference-only
    # files (default-excluded test files and files outside a file target's
    # duplicate-detection scope).
    files: int


@dataclass(frozen=True)
class UnitCounts:
    """Corpus size at two extraction stages, with a breakdown of the extracted units."""

    extracted: int
    semantic_eligible: int
    by_language: Mapping[str, int] = field(default_factory=dict)
    by_type: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_units(cls, units: Sequence[CodeUnit], *, semantic_eligible: int) -> UnitCounts:
        """Count extracted units by language and by unit type.

        ``by_type`` always carries all three :class:`CodeUnitType` names
        (``class``, ``function``, ``method``), with ``0`` for a type that is
        absent, because the enum is closed. ``by_language`` carries only the
        languages actually present, because the language set is open-ended.

        :param units: Extracted code units.
        :param semantic_eligible: Count of units eligible for semantic embedding.
        :return: Unit counts with both breakdowns populated.
        """
        by_language = dict(sorted(Counter(unit.language for unit in units).items()))
        type_counts = Counter(unit.unit_type.name.lower() for unit in units)
        by_type = {
            name: type_counts.get(name, 0)
            for name in sorted(member.name.lower() for member in CodeUnitType)
        }
        return cls(
            extracted=len(units),
            semantic_eligible=semantic_eligible,
            by_language=by_language,
            by_type=by_type,
        )


@dataclass(frozen=True)
class RunRecord:
    """Resolved configuration and scope actually applied by one analysis run."""

    tool_version: str
    root: Path
    # The analysis target as given, preserving an explicit symlink's own name.
    target: Path
    languages: tuple[str, ...] | None
    exclude_patterns: tuple[str, ...]
    respect_gitignore: bool
    include_private: bool
    include_stubs: bool
    extracted_files: int
    units: UnitCounts
    traditional: TraditionalSettings | None
    semantic: SemanticSettings | None
    unused: UnusedSettings | None

    @property
    def analysis_mode(self) -> AnalysisMode:
        """Return the analysis mode this run record implies.

        :return: ``"combined"`` when traditional and semantic both ran, else
            whichever of ``"traditional"``/``"semantic"``/``"unused"`` ran.
        """
        if self.traditional is not None and self.semantic is not None:
            return "combined"
        if self.traditional is not None:
            return "traditional"
        if self.semantic is not None:
            return "semantic"
        return "unused"


@dataclass(frozen=True)
class CheckRecord:
    """Status of one analysis check within a run."""

    status: CheckStatus
    files: int | None = None
    files_failed: int = 0
    diagnostics: int = 0


@dataclass(frozen=True)
class AnalysisChecks:
    """Per-check status for one completed analysis run."""

    extraction: CheckRecord
    traditional: CheckRecord
    semantic: CheckRecord
    unused: CheckRecord

    @property
    def incomplete_reasons(self) -> list[str]:
        """Return human-readable reasons the run is not ``complete``.

        :return: One reason per contributing check, empty when nothing degraded.
        """
        reasons: list[str] = []
        if self.extraction.status == "empty":
            reasons.append("no code units extracted")
        elif self.extraction.status == "partial":
            reasons.append(f"{self.extraction.files_failed} files with extraction errors")
        if self.semantic.status == "fallback":
            reasons.append("semantic analysis fell back to traditional results")
        if self.unused.status == "partial":
            reasons.append(f"{self.unused.files_failed} files skipped by unused analysis")
        return reasons

    @property
    def analysis_status(self) -> AnalysisStatus:
        """Return the overall status these checks imply.

        :return: ``"empty"`` when extraction produced no units, ``"partial"``
            when any check degraded, else ``"complete"``.
        """
        if self.extraction.status == "empty":
            return "empty"
        if self.incomplete_reasons:
            return "partial"
        return "complete"


def derive_checks(
    run: RunRecord,
    *,
    extracted_units: int,
    extraction_diagnostics: list[ExtractionDiagnostic],
    semantic_fallback: bool,
    semantic_diagnostics: list[ExtractionDiagnostic],
    unused_diagnostics: list[ExtractionDiagnostic],
) -> AnalysisChecks:
    """Derive per-check status from a run record and the diagnostics it produced.

    :param run: Resolved run record.
    :param extracted_units: Total code units extraction produced.
    :param extraction_diagnostics: Diagnostics raised during extraction.
    :param semantic_fallback: Whether combined mode fell back to traditional-only.
    :param semantic_diagnostics: Diagnostics raised during the semantic stage.
    :param unused_diagnostics: Diagnostics raised while building the unused reference graph.
    :return: Derived per-check status.
    """
    failed_files = {
        diagnostic.file_path
        for diagnostic in extraction_diagnostics
        if diagnostic.code in INCOMPLETE_EXTRACTION_CODES
    }
    if extracted_units == 0:
        extraction_status: CheckStatus = "empty"
    elif failed_files:
        extraction_status = "partial"
    else:
        extraction_status = "completed"
    extraction = CheckRecord(
        status=extraction_status,
        files=run.extracted_files,
        files_failed=len(failed_files),
        diagnostics=len(extraction_diagnostics),
    )

    traditional = CheckRecord(status="completed" if run.traditional is not None else "disabled")

    if run.semantic is None:
        semantic = CheckRecord(status="disabled")
    else:
        semantic = CheckRecord(
            status="fallback" if semantic_fallback else "completed",
            diagnostics=len(semantic_diagnostics),
        )

    if run.unused is None:
        unused = CheckRecord(status="disabled")
    else:
        unused = CheckRecord(
            status="partial" if unused_diagnostics else "completed",
            files=run.unused.files,
            files_failed=len(unused_diagnostics),
            diagnostics=len(unused_diagnostics),
        )

    return AnalysisChecks(
        extraction=extraction, traditional=traditional, semantic=semantic, unused=unused
    )


@dataclass(frozen=True)
class FocusSummary:
    """Scope ``--focus`` applied to one report, and what fell out of it.

    The complete result stays corpus-wide; only the findings a focused
    report emits change. ``out_of_focus_duplicates``/``out_of_focus_unused``
    are counted in finding units (an exact family counts once), so they add
    up with the focused report's own counts back to the unfocused totals.
    """

    paths: tuple[Path, ...]
    units: int
    out_of_focus_duplicates: int
    out_of_focus_unused: int


@dataclass
class AnalysisResult:
    """Full analysis result."""

    units: list[CodeUnit]
    traditional_duplicates: list[DuplicatePair]
    semantic_duplicates: list[DuplicatePair]
    hybrid_duplicates: list[HybridDuplicate]
    potentially_unused: list[CodeUnit]
    run: RunRecord
    semantic_fallback: bool = False
    semantic_fallback_reason: str | None = None
    extraction_diagnostics: list[ExtractionDiagnostic] = field(default_factory=list)
    semantic_diagnostics: list[ExtractionDiagnostic] = field(default_factory=list)
    unused_diagnostics: list[ExtractionDiagnostic] = field(default_factory=list)
    unused_supported_languages: tuple[str, ...] = ("python",)
    unused_excluded_units: int = 0
    suppressed_duplicates: int = 0
    suppressed_unused: int = 0
    embedding_stats: EmbeddingRunStats | None = None
    focus: FocusSummary | None = None

    @property
    def analysis_mode(self) -> AnalysisMode:
        """Return the analysis mode this result's run record implies.

        :return: ``"combined"``, ``"traditional"``, ``"semantic"``, or ``"unused"``.
        """
        return self.run.analysis_mode

    @property
    def checks(self) -> AnalysisChecks:
        """Return per-check status derived from this result's run record.

        :return: Derived per-check status.
        """
        return derive_checks(
            self.run,
            extracted_units=len(self.units),
            extraction_diagnostics=self.extraction_diagnostics,
            semantic_fallback=self.semantic_fallback,
            semantic_diagnostics=self.semantic_diagnostics,
            unused_diagnostics=self.unused_diagnostics,
        )

    @property
    def analysis_status(self) -> AnalysisStatus:
        """Return the overall status this result's derived checks imply.

        :return: ``"complete"``, ``"partial"``, or ``"empty"``.
        """
        return self.checks.analysis_status

    @property
    def all_duplicates(self) -> list[HybridDuplicate] | list[DuplicatePair]:
        """Return the available duplicate list for this analysis mode.

        :return: Hybrid duplicates in combined mode, otherwise traditional plus semantic pairs.
        """
        if self.analysis_mode == "combined":
            return self.hybrid_duplicates
        return self.traditional_duplicates + self.semantic_duplicates
