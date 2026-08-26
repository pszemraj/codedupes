"""Shared contracts for language-specific source extractors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from codedupes.models import CodeUnit, ExtractionDiagnostic


@dataclass(frozen=True)
class BackendResult:
    """Code units and diagnostics produced by parsing one file."""

    units: tuple[CodeUnit, ...]
    diagnostics: tuple[ExtractionDiagnostic, ...] = ()


class LanguageBackend(Protocol):
    """Contract implemented by non-Python language backends."""

    language: str
    dialect: str

    def extract_file(self, file_path: Path) -> BackendResult:
        """Parse one file and return code units plus extraction diagnostics."""
