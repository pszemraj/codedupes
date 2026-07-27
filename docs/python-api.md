# Python API

This page covers programmatic usage.
CLI flag defaults are documented in
[docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md); CLI JSON schemas/exit codes are
documented in [docs/output.md](https://github.com/pszemraj/codedupes/blob/main/docs/output.md).
Analysis behavior defaults are documented in
[docs/analysis-defaults.md](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md).
Semantic model aliases/profile defaults/task behavior are documented in
[docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md).
Device selection and MPS lifecycle behavior are documented in
[docs/accelerators.md](https://github.com/pszemraj/codedupes/blob/main/docs/accelerators.md).

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
    jaccard_threshold=0.85,
    semantic_threshold=None,  # resolves from model profile
    model_name="embeddinggemma-300m",
    semantic_task="semantic-similarity",
    device="auto",
    mps_fallback=None,
    mps_memory_fraction=None,
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
        device="auto",
    )
)

analyzer.analyze("./src")
hits = analyzer.search("load csv data", top_k=10)

for unit, score in hits:
    print(f"{score:.3f}", unit.qualified_name)
```

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

`mps_fallback` controls unsupported PyTorch operations. It does not disable the analyzer's explicit
OOM recovery. The shared model cache and inference path are serialized in-process; after an OOM
forces a model to CPU, that CPU placement remains cached until `clear_model_cache()` is called or a
different model/device key is requested.

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

- Call graph and unused detection are heuristic and conservative by default.
- `AnalyzerConfig` enforces workflow dependencies:
  - semantic-only settings require `run_semantic=True`
  - traditional-only settings require `run_traditional=True`
  - `strict_unused=True` requires `run_unused=True`
- Semantic candidate defaults and tiny-traditional filtering defaults are defined in
  [docs/analysis-defaults.md](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md).
- Semantic analysis may download model weights on first use.
- `device`, `mps_fallback`, and `mps_memory_fraction` require `run_semantic=True`.
- Model alias and profile-resolution behavior is documented in
  [docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md).
