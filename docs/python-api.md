# Python API

Use `analyze_directory` for a one-shot analysis or `CodeAnalyzer` when configuration and semantic search share one analyzed corpus.

## Quick Start

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

## Configurable Analyzer

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

`run_unused=True` remains valid for a mixed tree, but only Python units enter the reference graph. Traditional and semantic duplicate checking are same-language by default, with each language gated by its calibrated profile threshold when `semantic_threshold` is `None`; `AnalyzerConfig(cross_language=True)` (or `analyze_directory(..., cross_language=True)`) also reports uncalibrated cross-language semantic pairs at the looser of the two gates. Semantic query search remains cross-language.

## Semantic Query Search

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
    )
)

analyzer.index("./src")
hits = analyzer.search("load csv data", top_k=10, threshold=0.55)

for unit, score in hits:
    print(f"{score:.3f}", unit.qualified_name)
```

`search(query, top_k=10, threshold=None)` resolves its floor as `threshold`, else `config.semantic_threshold`, else the model profile's search default. Prefer the per-call `threshold`: it applies to that query only, while `config.semantic_threshold` also replaces every calibrated per-language duplicate gate with one flat value.

An unset `AnalyzerConfig.semantic_task` resolves by operation: `index()` uses `code-retrieval`, while `analyze()` uses `semantic-similarity`. An explicit task overrides either default, but a custom instruction prefix, alternate EmbeddingGemma task, alternate built-in revision, or non-default `trust_remote_code` requires an explicit threshold because the profile default was not calibrated in that embedding space. See [task defaults](model-profiles.md#semantic-task-defaults-and-choices).

`AnalyzerConfig.mode` declares which contract enforces that requirement. The default `mode="check"` rejects an uncalibrated context without `semantic_threshold` at construction, before any extraction or model load. A config that drives `index()`/`search()` must pass `mode="search"` instead: search thresholds are calibrated independently of the duplicate gates, so validation defers to query time (`search()` raises if the resolved context has no calibrated search default and no explicit threshold). `analyze()` rejects `mode="search"` configs.

`index()` extracts the corpus and computes (or loads from cache) its embeddings without the all-pairs duplicate scan, traditional analysis, or unused-code analysis that `analyze()` runs, so building a search corpus stays linear in corpus size. Prefer `index()` before search. A search after `analyze()` reuses the analysis task and therefore requires an explicit search threshold when that task changes the model's prompt or route, as it does for EmbeddingGemma.

Corpus units whose tokenized input exceeds the model's context window are skipped by both `index()` and `analyze()` rather than raising: they leave the embedding matrix and the searchable corpus, and each one is reported through `analyzer.semantic_diagnostics` (and `AnalysisResult.semantic_diagnostics` for `analyze()`) with code `semantic-context-overflow`. A `search()` query too long for the model still raises, because a truncated query has no result to omit.

```python
analyzer.index("./src")

for diagnostic in analyzer.semantic_diagnostics:
    print(diagnostic.code, diagnostic.file_path, diagnostic.message)
```

Each `index()` or `analyze()` call replaces the analyzer's corpus-specific state before extraction. `search()` therefore targets only the most recent run and requires it to have semantic embeddings. A later empty or nonsemantic analysis cannot reuse an older corpus accidentally. The analyzer also binds the matrix to its canonical model, resolved revision (a pinned commit, the requested revision label, or a local-directory content fingerprint), and vector-affecting runtime configuration. If any of those changes before a query—for example, local weights are replaced in place—`search()` requires a fresh `index()`/`analyze()` instead of comparing vectors from different coordinate systems. With `AnalyzerConfig(strict_revision_cache=True)`, an unpinned hub revision must instead resolve to a concrete commit; that commit is also the model-load key, so a moved branch cannot reuse the process's model instance from its previous commit. A search identity whose symbolic revision cannot be mapped offline fails closed; the default label keying always resolves. See [Embedding cache](caching.md#what-invalidates-what).

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
- `AnalysisResult.hybrid_duplicates`: synthesized default duplicate candidates; gated semantic-only matches use `semantic_high_confidence` when lexical and statement-count evidence corroborate them, otherwise `semantic_review`
- `AnalysisResult.traditional_duplicates`: raw traditional duplicates (diagnostics)
- `AnalysisResult.semantic_duplicates`: raw semantic duplicates (diagnostics)
- `AnalysisResult.potentially_unused`: Python-only heuristic unused candidates
- `AnalysisResult.extraction_diagnostics`: recoverable parser diagnostics and skipped-unit reasons
- `AnalysisResult.semantic_diagnostics`: units the semantic stage skipped, mirroring `CodeAnalyzer.semantic_diagnostics` for that run
- `AnalysisResult.unused_excluded_units`: non-Python units intentionally excluded from unused analysis
- `AnalysisResult.unused_supported_languages`: languages the unused heuristic evaluates (currently always `("python",)`)
- `AnalysisResult.all_duplicates`: hybrid duplicates in combined mode; raw duplicates in single-method mode
- `AnalysisResult.analysis_mode`: `"combined"`, `"traditional"`, `"semantic"`, or `"none"`
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
