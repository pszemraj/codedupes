# Python API

Use `analyze_directory` for a one-shot analysis or `CodeAnalyzer` when configuration and semantic search share one analyzed corpus.

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
    min_semantic_statements=1,
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
        device="auto",
    )
)

analyzer.index("./src")
hits = analyzer.search("load csv data", top_k=10)

for unit, score in hits:
    print(f"{score:.3f}", unit.qualified_name)
```

An unset `AnalyzerConfig.semantic_task` resolves by operation: `index()` uses `code-retrieval`, while `analyze()` uses `semantic-similarity`. An explicit task overrides either default. See [task defaults](model-profiles.md#semantic-task-defaults-and-choices).

`index()` extracts the corpus and computes (or loads from cache) its embeddings without the all-pairs duplicate scan, traditional analysis, or unused-code analysis that `analyze()` runs, so building a search corpus stays linear in corpus size. `search()` also works after an `analyze()` run with semantic analysis enabled, when duplicate results and search share one corpus and therefore use the analysis task that produced those embeddings.

Each `index()` or `analyze()` call replaces the analyzer's corpus-specific state before extraction. `search()` therefore targets only the most recent run and requires it to have semantic embeddings. A later empty or nonsemantic analysis cannot reuse an older corpus accidentally. The analyzer also binds the matrix to its canonical model, resolved revision (a pinned commit, the requested revision label, or a local-directory content fingerprint), and vector-affecting runtime configuration. If any of those changes before a query—for example, local weights are replaced in place—`search()` requires a fresh `index()`/`analyze()` instead of comparing vectors from different coordinate systems. With `AnalyzerConfig(strict_revision_cache=True)`, an unpinned hub revision must instead resolve to a concrete commit, and a search identity whose symbolic revision cannot be mapped offline fails closed; the default label keying always resolves. See [Embedding cache](caching.md#what-invalidates-what).

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

See [Accelerators](accelerators.md) for unsupported-operator fallback, OOM recovery, and cached model placement. Long-lived processes can explicitly release the model:

```python
from codedupes.semantic import clear_model_cache

clear_model_cache()
```

An `mps_memory_fraction` override is also process-global. A later analyzer that uses the default `None` restores PyTorch's environment/default high-watermark ratio before its next MPS load; clearing the model alone does not reset the allocator cap.

## Logging

Model loading quiets known-noisy dependency loggers (httpx request lines, transformers/sentence-transformers chatter) automatically, but only ones still inheriting the root level — any logger you configure explicitly is left alone. To pin them yourself, or to a different level:

```python
from codedupes import quiet_dependency_loggers

quiet_dependency_loggers()  # or quiet_dependency_loggers(logging.ERROR)
```

## Key Result Types

- `AnalysisResult.units`: extracted functions, methods, and classes
- `AnalysisResult.hybrid_duplicates`: synthesized default duplicate candidates
- `AnalysisResult.traditional_duplicates`: raw traditional duplicates (diagnostics)
- `AnalysisResult.semantic_duplicates`: raw semantic duplicates (diagnostics)
- `AnalysisResult.potentially_unused`: heuristic unused candidates
- `AnalysisResult.all_duplicates`: hybrid duplicates in combined mode; raw duplicates in single-method mode
- `AnalysisResult.analysis_mode`: `"combined"`, `"traditional"`, `"semantic"`, or `"none"`
- `CodeUnit.uid`: definition identity (`path` and qualified name)

## Notes

- `AnalyzerConfig` enforces workflow dependencies:
  - semantic-only settings require `run_semantic=True`
  - traditional-only settings require `run_traditional=True`
  - `strict_unused=True` requires `run_unused=True`
- `device`, `mps_fallback`, and `mps_memory_fraction` require `run_semantic=True`. `embedding_cache=False` is accepted when semantic analysis is disabled and has no effect.
- [Analysis defaults](analysis-defaults.md) covers candidate scope and filtering.
- [Embedding cache](caching.md) covers persistent cache behavior.
- [Model profiles](model-profiles.md) covers aliases, thresholds, revisions, and task behavior.
