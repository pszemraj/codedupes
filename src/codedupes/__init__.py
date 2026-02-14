"""
codedupes - Detect duplicate and unused Python code.

Uses dual-approach detection:
1. Traditional: AST hashing, token hashing, Jaccard similarity
2. Semantic: Code embedding similarity via C2LLM (codefuse-ai/C2LLM-0.5B)

Example:
    from codedupes import analyze_directory

    result = analyze_directory("./src")

    for dup in result.semantic_duplicates:
        print(f"{dup.unit_a.name} ~ {dup.unit_b.name} ({dup.similarity:.0%})")

    for unused in result.potentially_unused:
        print(f"Unused: {unused.qualified_name}")
"""

from .analyzer import AnalyzerConfig, CodeAnalyzer, analyze_directory
from .models import AnalysisResult, CodeUnit, CodeUnitType, DuplicatePair

__version__ = "0.2.0"

__all__ = [
    "AnalysisResult",
    "AnalyzerConfig",
    "CodeAnalyzer",
    "CodeUnit",
    "CodeUnitType",
    "DuplicatePair",
    "analyze_directory",
]
