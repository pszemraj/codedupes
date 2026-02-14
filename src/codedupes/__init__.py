"""
codedupes - Detect duplicate and unused Python code.

Uses dual-approach detection:
1. Traditional: AST hashing, token hashing, Jaccard similarity
2. Semantic: Code embedding similarity via C2LLM (codefuse-ai/C2LLM-0.5B)

Example:
    from codedupes import analyze_directory

    result = analyze_directory("./src")

    for dup in result.hybrid_duplicates:
        print(f"{dup.unit_a.name} ~ {dup.unit_b.name} ({dup.confidence:.0%}, {dup.tier})")

    for unused in result.potentially_unused:
        print(f"Unused: {unused.qualified_name}")
"""

from .analyzer import AnalyzerConfig, CodeAnalyzer, analyze_directory
from .models import AnalysisResult, CodeUnit, CodeUnitType, DuplicatePair, HybridDuplicate

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0+unknown"
    __version_tuple__ = (0, 0, 0, "+unknown")

__all__ = [
    "AnalysisResult",
    "AnalyzerConfig",
    "CodeAnalyzer",
    "CodeUnit",
    "CodeUnitType",
    "DuplicatePair",
    "HybridDuplicate",
    "__version__",
    "__version_tuple__",
    "analyze_directory",
]
