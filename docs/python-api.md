# Python API

This page covers programmatic usage.
CLI flag defaults are documented in
[docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md); CLI JSON schemas/exit codes are
documented in [docs/output.md](https://github.com/pszemraj/codedupes/blob/main/docs/output.md).
Analysis behavior defaults are documented in
[docs/analysis-defaults.md](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md).
Semantic model aliases/profile defaults/task behavior are documented in
[docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md).

## Quick Start

```python
from codedupes import analyze_directory

result = analyze_directory(
    "./src",
    semantic_threshold=None,  # use model-profile default
    traditional_threshold=0.85,
    model_name="gte-modernbert-base",
    semantic_task="semantic-similarity",
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
    jaccard_threshold=0.85,
    semantic_threshold=None,  # resolves from model profile
    model_name="embeddinggemma-300m",
    semantic_task="semantic-similarity",
    run_traditional=True,
    run_semantic=True,
    run_unused=True,
    strict_unused=False,
    include_private=True,
    min_semantic_lines=3,
    semantic_unit_types=("function", "method"),
    filter_tiny_traditional=True,
    tiny_unit_statement_cutoff=3,
    tiny_near_jaccard_min=0.93,
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
    )
)

analyzer.analyze("./src")
hits = analyzer.search("load csv data", top_k=10)

for unit, score in hits:
    print(f"{score:.3f}", unit.qualified_name)
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

- Call graph and unused detection are heuristic and conservative by default.
- `AnalyzerConfig` enforces workflow dependencies:
  - semantic-only settings require `run_semantic=True`
  - traditional-only settings require `run_traditional=True`
  - `strict_unused=True` requires `run_unused=True`
- Semantic candidate defaults and tiny-traditional filtering defaults are defined in
  [docs/analysis-defaults.md](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md).
- Semantic analysis may download model weights on first use.
- Model alias and profile-resolution behavior is documented in
  [docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md).
