"""Data models for extracted code units."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Literal

from codedupes.pairs import ordered_pair_key


class CodeUnitType(Enum):
    """Kinds of analyzed code units."""

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

    # Computed on demand
    _ast_hash: str | None = field(default=None, repr=False)
    _token_hash: str | None = field(default=None, repr=False)

    # For reference-graph / usage analysis
    module_name: str = ""  # root-relative module identity
    import_module_name: str = ""  # importable module identity (may omit a source root)
    referenced_names: set[str] = field(default_factory=set)  # names this unit references
    resolved_referenced_names: set[str] = field(default_factory=set)  # proven identities
    referenced_attributes: set[str] = field(default_factory=set)  # unresolved attribute tails
    module_attribute_references: set[str] = field(default_factory=set)  # module-rooted paths
    references: set[str] = field(default_factory=set)  # uids of units referencing this unit

    # API exposure markers
    is_public: bool = False
    is_dunder: bool = False
    is_exported: bool = False  # in __all__

    # Framework/runtime dispatch markers used by conservative unused analysis.
    is_dynamic_dispatch_hook: bool = False

    @property
    def uid(self) -> str:
        """Build a stable unique identifier for this code unit.

        The source line disambiguates legal same-scope redefinitions that share
        a file path and qualified name. Pair synthesis and reference bookkeeping
        require physical definitions to remain distinct.

        :return: ``"<path>::<qualified_name>@<lineno>"``.
        """
        return f"{self.file_path}::{self.qualified_name}@{self.lineno}"

    @property
    def is_likely_api(self) -> bool:
        """Indicate whether this unit is likely public API surface.

        :return: ``True`` if the unit is likely intentionally exposed.
        """
        return (
            self.is_exported
            or self.is_dunder
            or (self.is_public and self.unit_type == CodeUnitType.CLASS)
            or self.name in ("__init__", "__new__", "__call__")
        )


class _PairIdentityMixin:
    """Equality and hashing on the unordered unit pair, ignoring score fields.

    Same-type comparison only: pair classes never compare equal across types.
    Subclasses must be declared with ``@dataclass(eq=False)`` — dataclass-generated
    equality would otherwise override these inherited dunders.
    """

    unit_a: CodeUnit
    unit_b: CodeUnit

    def __hash__(self) -> int:
        return hash(ordered_pair_key(self.unit_a, self.unit_b))

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return ordered_pair_key(self.unit_a, self.unit_b) == ordered_pair_key(
            other.unit_a, other.unit_b
        )


@dataclass(eq=False)
class DuplicatePair(_PairIdentityMixin):
    """A pair of code units identified as duplicates."""

    unit_a: CodeUnit
    unit_b: CodeUnit
    similarity: float
    method: str  # "ast_hash", "token_hash", "semantic"


HybridTier = Literal[
    "exact",
    "traditional_near",
    "hybrid_confirmed",
    "semantic_high_confidence",
]

AnalysisMode = Literal["combined", "traditional", "semantic", "none"]


@dataclass(eq=False)
class HybridDuplicate(_PairIdentityMixin):
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
    traditional_duplicates: list[DuplicatePair]  # AST/token/jaccard matches
    semantic_duplicates: list[DuplicatePair]  # Embedding similarity
    hybrid_duplicates: list[HybridDuplicate]  # Final combined output candidates
    potentially_unused: list[CodeUnit]  # No references, not API
    analysis_mode: AnalysisMode  # How duplicates were synthesized
    filtered_raw_duplicates: int = 0
    semantic_fallback: bool = False
    semantic_fallback_reason: str | None = None

    @property
    def all_duplicates(self) -> list[HybridDuplicate] | list[DuplicatePair]:
        """Return the available duplicate list for this analysis mode.

        :return: Duplicate list for the selected analysis mode.
        """
        if self.analysis_mode == "combined":
            return self.hybrid_duplicates
        return self.traditional_duplicates + self.semantic_duplicates
