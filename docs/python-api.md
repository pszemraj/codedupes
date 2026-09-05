# Python API

Use `analyze_directory` for a one-shot analysis or `CodeAnalyzer` when configuration and semantic search share one analyzed corpus.

## Quick start

```python
from codedupes import analyze_directory

result = analyze_directory(
    "./src",
    semantic_threshold=None,  # use the profile's calibrated per-language gates
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

## Configurable analyzer

```python
from codedupes import AnalyzerConfig, CodeAnalyzer

config = AnalyzerConfig(
    run_traditional=True,
    run_semantic=True,
    run_unused=False,
    semantic_unit_types=("function", "method", "class"),
    min_semantic_statements=1,
    languages=("rust", "typescript"),
)

analyzer = CodeAnalyzer(config)
result = analyzer.analyze("./src")
```

## Language selection and extraction diagnostics

Omit `languages` to auto-detect every supported source type, or pass canonical names/aliases through `AnalyzerConfig`:

```python
from codedupes import AnalyzerConfig, CodeAnalyzer

analyzer = CodeAnalyzer(
    AnalyzerConfig(
        languages=("python", "rs", "ts"),
        run_traditional=True,
        run_semantic=False,
        run_unused=True,
    )
)
result = analyzer.analyze(".")

for diagnostic in result.extraction_diagnostics:
    print(diagnostic.code, diagnostic.language, diagnostic.file_path, diagnostic.message)

print("non-Python units excluded from unused analysis:", result.unused_excluded_units)
```

`run_unused=True` remains valid for a mixed tree; see [unused analysis scope](analysis-defaults.md#potentially-unused-defaults). Pass `AnalyzerConfig(cross_language=True)` or `analyze_directory(..., cross_language=True)` to opt into [cross-language duplicate pairs](analysis-defaults.md#semantic-duplicate-gate-defaults).

## Semantic query search

```python
from codedupes import AnalyzerConfig, CodeAnalyzer

analyzer = CodeAnalyzer(
    AnalyzerConfig(
        mode="search",
        run_traditional=False,
        run_semantic=True,
        run_unused=False,
        model_name="gte-modernbert-base",
        device="auto",
        search_document="contextual",
    )
)

analyzer.index("./src")
hits = analyzer.search("load csv data", top_k=10, threshold=0.55)

print("extracted:", analyzer.extracted_unit_count)
for unit, score in hits:
    print(f"{score:.3f}", unit.qualified_name)
```

`search(query, top_k=10, threshold=None)` resolves its floor as `threshold`, else `config.semantic_threshold`, else the model profile's search default. Prefer the per-call `threshold`: it applies to that query only, while `config.semantic_threshold` also replaces every calibrated per-language duplicate gate with one flat value.

See [task defaults and calibration requirements](model-profiles.md#semantic-task-defaults-and-choices) before overriding `semantic_task`, the prompt, revision, or remote-code setting.

`AnalyzerConfig.mode` declares which contract enforces that requirement. The default `mode="check"` rejects an uncalibrated context without `semantic_threshold` at construction, before any extraction or model load. `index()` and `search()` accept either mode. For a search-only workflow, use `mode="search"` to defer calibration validation to query time (`search()` raises if the resolved context has no calibrated search default and no explicit threshold). `analyze()` rejects `mode="search"` configs.

`index()` extracts the corpus and computes (or loads from cache) its embeddings without the all-pairs duplicate scan, traditional analysis, or unused-code analysis that `analyze()` runs, so building a search corpus stays linear in corpus size. Prefer `index()` before search. `analyzer.extracted_unit_count` reports the pre-filter extraction count from the latest `index()` or `analyze()` run, which can be larger than the count returned by `index()` after semantic eligibility and context-window filtering. A search after `analyze()` reuses the analysis task and therefore requires an explicit search threshold when that task changes the model's prompt or route, as it does for EmbeddingGemma.

`AnalyzerConfig.search_document` is `"source"` by default, preserving the calibrated source-only score distribution. `"contextual"` prepends language, root-relative path, and qualified symbol to each search document before the code. Contextual mode makes paths and symbols available to retrieval but changes the input distribution; the source-only thresholds have not been calibrated for it. Searching a contextual index requires an explicit `search(threshold=...)` or `config.semantic_threshold`; tune that value against representative queries. This requirement follows the indexed representation even if the config is changed afterward. Its [cache behavior](caching.md#what-invalidates-what) follows the complete document input. `analyze()` always embeds bare source for duplicate detection regardless of this search-only setting.

Units skipped by the [context-window policy](analysis-defaults.md#semantic-candidate-defaults) are reported through `analyzer.semantic_diagnostics` and, for `analyze()`, `AnalysisResult.semantic_diagnostics`:

```python
analyzer.index("./src")

for diagnostic in analyzer.semantic_diagnostics:
    print(diagnostic.code, diagnostic.file_path, diagnostic.message)
```

Each `index()` or `analyze()` call replaces the analyzer's corpus-specific state before extraction. `search()` therefore targets only the most recent run and requires it to have semantic embeddings. A later empty or nonsemantic analysis cannot reuse an older corpus accidentally. The analyzer binds the matrix to its model, revision, and vector-affecting runtime configuration. If any of those changes before a query, `search()` requires a fresh `index()`/`analyze()`. Set `AnalyzerConfig(strict_revision_cache=True)` for [strict revision resolution](caching.md#what-invalidates-what).

## Progress and embedding telemetry

`AnalyzerConfig.progress` accepts `"auto"` (default), `"always"`, or `"never"`. Auto mode renders embedding progress only for more than 100 uncached inputs when stderr is a TTY. The same keyword is available on `compute_embeddings`, `compute_embeddings_with_identity`, `run_semantic_analysis`, and `run_semantic_analysis_with_identity`.

The low-level functions accept an `EmbeddingRunStats` collector through `stats=` and fill it in place. `AnalysisResult.embedding_stats` contains that collector after successful semantic analysis; `CodeAnalyzer.embedding_stats` exposes it after `index()`. Both are `None` when semantic work did not run, failed, or fell back. Non-fatal persistent-cache failures observed during corpus or query embedding are appended to `cache_warnings`.

```python
from pathlib import Path
from codedupes.extractor import CodeExtractor
from codedupes.semantic import EmbeddingRunStats, compute_embeddings

repo_root = Path("./src").resolve()
units = CodeExtractor(repo_root).extract_all()
stats = EmbeddingRunStats()
embeddings = compute_embeddings(
    units,
    cache_scope=repo_root,
    progress="never",
    stats=stats,
)
print(stats.cache_hit_rows, stats.encoded_inputs, stats.model_loaded)
```

See [embedding telemetry](output.md#embedding-telemetry) for field definitions. Low-level `compute_embeddings*` calls require `cache_scope` for persistent reuse and do not publish corpus manifests; use `CodeAnalyzer` for move/deletion tracking. Without `overflow_report`, an over-context input raises. Supplying that collector drops its rows, so callers must remove the reported units before using the returned matrix. `CodeAnalyzer` maintains that alignment automatically.

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

See [MPS allocator policy](accelerators.md#mps-memory-policy-and-oom-recovery) for the process-wide effects of `mps_memory_fraction`.

## Logging

Model loading quiets known-noisy dependency loggers (httpx request lines, transformers/sentence-transformers chatter) automatically, but only ones still inheriting the root level - any logger you configure explicitly is left alone. To pin them yourself, or to a different level:

```python
from codedupes import quiet_dependency_loggers

quiet_dependency_loggers()  # or quiet_dependency_loggers(logging.ERROR)
```

## Key result types

- `AnalysisResult.units`: extracted functions, methods, and classes
- `AnalysisResult.hybrid_duplicates`: synthesized duplicate candidates with [confidence tiers](analysis-defaults.md#hybrid-synthesis-confidence-defaults)
- `AnalysisResult.traditional_duplicates`: raw traditional duplicates (diagnostics)
- `AnalysisResult.semantic_duplicates`: raw semantic duplicates (diagnostics)
- `AnalysisResult.potentially_unused`: Python-only heuristic unused candidates
- `AnalysisResult.extraction_diagnostics`: recoverable parser diagnostics and skipped-unit reasons
- `AnalysisResult.semantic_diagnostics`: units the semantic stage skipped, mirroring `CodeAnalyzer.semantic_diagnostics` for that run
- `AnalysisResult.unused_excluded_units`: non-Python units intentionally excluded from unused analysis
- `AnalysisResult.unused_supported_languages`: languages the unused heuristic evaluates (currently always `("python",)`)
- `AnalysisResult.all_duplicates`: hybrid duplicates in combined mode; raw duplicates in single-method mode
- `AnalysisResult.analysis_mode`: `"combined"`, `"traditional"`, `"semantic"`, or `"none"`
- `AnalysisResult.embedding_stats`: `EmbeddingRunStats` for a successful semantic corpus call, otherwise `None`
- `CodeUnit.uid`: in-run definition identity, `<path>::<language>::<qualified name>::<start byte>` for every language; the byte position keeps overloads and redefinitions distinct
- `CodeUnit.language`, `dialect`, and `native_kind`: canonical language plus parser-specific syntax kind
- `CodeUnit.start_byte`/`end_byte`: exact byte range used to slice the emitted source
- `CodeUnit.structural_hash`, `identifiers`, and `statement_count`: backend-computed language-neutral features

## Notes

- `AnalyzerConfig` enforces workflow dependencies:
  - semantic-only settings require `run_semantic=True`
  - traditional-only settings require `run_traditional=True`
  - `strict_unused=True` requires `run_unused=True`
- `device`, `mps_fallback`, and `mps_memory_fraction` require `run_semantic=True`. `embedding_cache=False` is accepted when semantic analysis is disabled and has no effect.
- [Analysis defaults](analysis-defaults.md) covers candidate scope and filtering.
- [Embedding cache](caching.md) covers persistent cache behavior.
- [Model profiles](model-profiles.md) covers aliases, thresholds, revisions, and task behavior.
