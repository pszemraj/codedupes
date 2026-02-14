"""Data models for extracted code units."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path


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
        # Unordered pair
        return hash(frozenset([self.unit_a.uid, self.unit_b.uid]))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DuplicatePair):
            return False
        return {self.unit_a.uid, self.unit_b.uid} == {other.unit_a.uid, other.unit_b.uid}


@dataclass
class AnalysisResult:
    """Full analysis result."""

    units: list[CodeUnit]
    exact_duplicates: list[DuplicatePair]  # AST/token hash matches
    semantic_duplicates: list[DuplicatePair]  # Embedding similarity
    potentially_unused: list[CodeUnit]  # No references, not API

    @property
    def all_duplicates(self) -> list[DuplicatePair]:
        seen = set()
        result = []
        for dup in self.exact_duplicates + self.semantic_duplicates:
            if dup not in seen:
                seen.add(dup)
                result.append(dup)
        return result
