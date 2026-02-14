# Python API

## Quick Start

```python
from codedupes import analyze_directory

result = analyze_directory(
    "./src",
    semantic_threshold=0.82,
    traditional_threshold=0.85,
)

for dup in result.exact_duplicates:
    print(dup.unit_a.qualified_name, "<->", dup.unit_b.qualified_name, dup.method)

for dup in result.semantic_duplicates:
    print(dup.unit_a.qualified_name, "<->", dup.unit_b.qualified_name, dup.similarity)

for unit in result.potentially_unused:
    print("Unused:", unit.qualified_name)
```

## Configurable Analyzer

```python
from codedupes import AnalyzerConfig, CodeAnalyzer

config = AnalyzerConfig(
    jaccard_threshold=0.85,
    semantic_threshold=0.82,
    run_traditional=True,
    run_semantic=True,
    run_unused=True,
    strict_unused=False,
    include_private=True,
    min_semantic_lines=3,
)

analyzer = CodeAnalyzer(config)
result = analyzer.analyze("./src")
```

## Semantic Query Search

```python
from codedupes import AnalyzerConfig, CodeAnalyzer

analyzer = CodeAnalyzer(
    AnalyzerConfig(
        run_traditional=False,
        run_semantic=True,
        run_unused=False,
    )
)

analyzer.analyze("./src")
hits = analyzer.search("load csv data", top_k=10)

for unit, score in hits:
    print(f"{score:.3f}", unit.qualified_name)
```

## Key Result Types

- `AnalysisResult.units`: extracted functions, methods, and classes
- `AnalysisResult.exact_duplicates`: exact and near-traditional duplicates
- `AnalysisResult.semantic_duplicates`: embedding-similarity duplicates
- `AnalysisResult.potentially_unused`: heuristic unused candidates

## Notes

- Call graph and unused detection are heuristic and conservative by default.
- Semantic analysis may download model weights on first use.
