# Python API

Use `analyze_directory` for a one-shot analysis or `CodeAnalyzer` when configuration and semantic
search share one analyzed corpus.

## Quick Start

```python
from codedupes import analyze_directory

result = analyze_directory(
    "./src",
    semantic_threshold=None,  # use model-profile default
    traditional_threshold=0.85,
    model_name="gte-modernbert-base",
    semantic_task="semantic-similarity",
    device="auto",
)

for dup in result.hybrid_duplicates:
    print(
        dup.unit_a.qualified_name,
        "<->",
        dup.unit_b.qualified_name,
        dup.tier,
        f"{dup.confidence:.2f}",
    )

for unit in result.potentially_unused:
    print("Unused:", unit.qualified_name)
```

## Configurable Analyzer

```python
from codedupes import AnalyzerConfig, CodeAnalyzer

config = AnalyzerConfig(
    run_traditional=True,
    run_semantic=True,
    run_unused=False,
    semantic_unit_types=("function", "method", "class"),
    min_semantic_lines=1,
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
        model_name="gte-modernbert-base",
        semantic_task="code-retrieval",
        device="auto",
    )
)

analyzer.analyze("./src")
hits = analyzer.search("load csv data", top_k=10)

for unit, score in hits:
    print(f"{score:.3f}", unit.qualified_name)
```

Each `analyze()` call replaces the analyzer's corpus-specific state before
extraction. `search()` therefore targets only the most recent analysis and
requires that run to have semantic embeddings. A later empty or nonsemantic
analysis cannot reuse an older corpus accidentally.

## Apple Silicon configuration

Use an explicit device for validation and set an allocator cap only when needed:

```python
from codedupes import AnalyzerConfig, CodeAnalyzer

analyzer = CodeAnalyzer(
    AnalyzerConfig(
        device="mps",
        mps_fallback=True,
        mps_memory_fraction=0.9,
        batch_size=4,
    )
)
result = analyzer.analyze("./src")
```

See [Accelerators](accelerators.md) for unsupported-operator fallback, OOM recovery, and cached model
placement. Long-lived processes can explicitly release the model:

```python
from codedupes.semantic import clear_model_cache

clear_model_cache()
```

## Key Result Types

- `AnalysisResult.units`: extracted functions, methods, and classes
- `AnalysisResult.hybrid_duplicates`: synthesized default duplicate candidates
- `AnalysisResult.traditional_duplicates`: raw traditional duplicates (diagnostics)
- `AnalysisResult.semantic_duplicates`: raw semantic duplicates (diagnostics)
- `AnalysisResult.potentially_unused`: heuristic unused candidates
- `AnalysisResult.all_duplicates`: hybrid duplicates in combined mode; raw duplicates in single-method mode
- `AnalysisResult.analysis_mode`: `"combined"`, `"traditional"`, `"semantic"`, or `"none"`

## Notes

- `AnalyzerConfig` enforces workflow dependencies:
  - semantic-only settings require `run_semantic=True`
  - traditional-only settings require `run_traditional=True`
  - `strict_unused=True` requires `run_unused=True`
- `device`, `mps_fallback`, `mps_memory_fraction`, and `embedding_cache` require `run_semantic=True`.
- [Analysis defaults](analysis-defaults.md) covers candidate scope and filtering.
- [Embedding cache](caching.md) covers persistent cache behavior.
- [Model profiles](model-profiles.md) covers aliases, thresholds, revisions, and task behavior.
