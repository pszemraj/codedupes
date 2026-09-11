"""
codedupes - Detect duplicate source code and conservative Python dead-code candidates.

Uses dual-approach detection:
1. Traditional: structural hashing, token hashing, Jaccard similarity
2. Semantic: Code embedding similarity via model profiles (default gte-modernbert-base)

Example:
    from codedupes import analyze_directory

    result = analyze_directory("./src")

    for dup in result.hybrid_duplicates:
        print(f"{dup.unit_a.name} ~ {dup.unit_b.name} ({dup.confidence:.0%}, {dup.tier})")

    for unused in result.potentially_unused:
        print(f"Unused: {unused.qualified_name}")
"""

from .analyzer import AnalyzerConfig, CodeAnalyzer, analyze_directory
from .logging_utils import quiet_dependency_loggers
from .models import (
    HYBRID_TIERS,
    AnalysisResult,
    CodeUnit,
    CodeUnitType,
    DuplicatePair,
    ExtractionDiagnostic,
    HybridDuplicate,
)
from .report import (
    ACTIONABLE_TIERS,
    ReportPolicy,
    ReportSelection,
    check_result_to_json,
    run_should_fail,
    search_result_to_json,
    select_findings,
    to_json_text,
    withheld_only_failure,
)

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0+unknown"
    __version_tuple__ = (0, 0, 0, "+unknown")

__all__ = [
    "ACTIONABLE_TIERS",
    "HYBRID_TIERS",
    "AnalysisResult",
    "AnalyzerConfig",
    "CodeAnalyzer",
    "CodeUnit",
    "CodeUnitType",
    "DuplicatePair",
    "ExtractionDiagnostic",
    "HybridDuplicate",
    "ReportPolicy",
    "ReportSelection",
    "__version__",
    "__version_tuple__",
    "analyze_directory",
    "check_result_to_json",
    "quiet_dependency_loggers",
    "run_should_fail",
    "search_result_to_json",
    "select_findings",
    "to_json_text",
    "withheld_only_failure",
]
