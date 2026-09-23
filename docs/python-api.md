# Python API

Use `analyze_directory` for a one-shot analysis or `CodeAnalyzer` when configuration and semantic search share one analyzed corpus. See [analysis defaults](analysis-defaults.md#what-a-default-check-does) for the stages enabled by default and [installation](install.md) before running these examples. Paths are relative to the process's working directory.

Use `AnalyzerConfig(run_semantic=False)` when you need a no-model traditional/unused analysis. Use `AnalyzerConfig(mode="search", ...)` when the workflow is indexing and querying code rather than reporting duplicate pairs.

## Quick start

```python
from codedupes import analyze_directory

result = analyze_directory(
    "./src",
    traditional_threshold=0.85,
)

print(f"Analyzed {len(result.units)} code units")

for dup in result.hybrid_duplicates:
    print(
        dup.unit_a.qualified_name,
        "<->",
        dup.unit_b.qualified_name,
        dup.tier,
        f"{dup.score:.2f}",
    )

for unit in result.potentially_unused:
    print("Unused:", unit.qualified_name)
```

## Configurable analyzer

This example includes classes and one-statement units in semantic comparison and limits extraction to Rust and TypeScript:

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

### `AnalyzerConfig` field map

| Area | Fields | Behavior |
| --- | --- | --- |
| Extraction | `exclude_patterns`, `respect_gitignore`, `include_private`, `languages`, `include_stubs` | [Scope defaults](analysis-defaults.md#extraction-scope-defaults) and [language selection](polyglot-languages.md#supported-files) |
| Analysis stages | `mode`, `run_traditional`, `run_semantic`, `run_unused`, `strict_unused`, `allow_semantic_fallback`, `suppress_test_semantic_matches` | [Check defaults](analysis-defaults.md), [fallback behavior](output.md#exit-codes), and [CLI option mapping](cli.md#codedupes-check-path) |
| Traditional matching | `jaccard_threshold`, `filter_tiny_traditional`, `tiny_unit_statement_cutoff` | [Traditional defaults](analysis-defaults.md#traditional-duplicate-defaults) |
| Semantic matching | `semantic_threshold`, `threshold_profile`, `cross_language`, `min_semantic_statements`, `semantic_unit_types`, `semantic_task` | [Semantic candidates and gates](analysis-defaults.md#semantic-duplicate-gate-defaults) and [model profiles](model-profiles.md) |
| Model runtime | `model_name`, `instruction_prefix`, `model_revision`, `trust_remote_code`, `device`, `mps_fallback`, `mps_memory_fraction`, `batch_size` | [Model loading](model-profiles.md) and [accelerator behavior](accelerators.md) |
| Cache, progress, and search | `embedding_cache`, `strict_revision_cache`, `progress`, `search_document` | [Cache controls](caching.md), [telemetry](#progress-and-embedding-telemetry), and [query search](#semantic-query-search) |

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

`CodeAnalyzer.search()` returns code-unit/score pairs. For file grouping, use [report helpers](#report-selection-and-json) or the [CLI's `--result-level file`](cli.md#codedupes-search-path-query) option.

For code search, call `index()` once, then call `search()` as many times as needed on that analyzer. The default source-only index has a calibrated profile threshold, so the basic workflow needs no threshold tuning:

```python
from codedupes import AnalyzerConfig, CodeAnalyzer

analyzer = CodeAnalyzer(
    AnalyzerConfig(
        mode="search",
        run_traditional=False,
        run_unused=False,
    )
)

analyzer.index("./src")
hits = analyzer.search("load csv data")

print("extracted:", analyzer.extracted_unit_count)
for unit, score in hits:
    print(f"{score:.3f}", unit.qualified_name)
```

Inspect `analyzer.extraction_diagnostics` for recoverable parse errors after indexing and `analyzer.semantic_diagnostics` for semantic-stage diagnostics. An empty result can mean no eligible definitions or no scores above the threshold; it does not by itself establish that every file was parsed successfully.

[Long-input diagnostics](analysis-defaults.md#semantic-candidate-defaults) remain available through `analyzer.semantic_diagnostics`; low-level `compute_embeddings*` and `run_semantic_analysis*` callers can collect them through `diagnostics=`.

`search(query, top_k=10, threshold=None)` resolves its floor as `threshold`, then `config.semantic_threshold`, then the selected threshold profile's search default. Prefer the per-call value when tuning one query: `config.semantic_threshold` also replaces every calibrated per-language duplicate gate with one flat value. Per-call thresholds must be finite; `NaN` and infinity raise `ValueError`, including for empty corpora and cached queries. Zero and finite negative floors are supported. After a search over a nonempty index, `analyzer.query_execution` records the effective threshold alongside each query vector's device and cache status.

`AnalyzerConfig`, `analyze_directory()`, `semantic.resolve_search_threshold()`, and `semantic.find_similar_to_query()` accept the [threshold profile choices](model-profiles.md#choosing-threshold-defaults). Numeric thresholds take precedence.

Set `search_document="contextual"` only when paths and symbol names should influence retrieval. It changes each document's input, so it requires an explicit `search(threshold=...)` or `semantic_threshold`; tune that threshold against representative queries.

See [task defaults and calibration requirements](model-profiles.md#semantic-task-defaults-and-choices) before overriding `semantic_task`, the prompt, revision, or remote-code setting.

`AnalyzerConfig.mode` declares which contract enforces that requirement. The default `mode="check"` rejects an uncalibrated context without `semantic_threshold` at construction, before any extraction or model load. `index()` and `search()` accept either mode. For a search-only workflow, use `mode="search"` to defer calibration validation to query time (`search()` raises if the resolved context has no calibrated search default and no explicit threshold). `analyze()` rejects `mode="search"` configs.

`index()` extracts the corpus and computes (or loads from cache) its embeddings without the all-pairs duplicate scan, traditional analysis, or unused-code analysis that `analyze()` runs, so building a search corpus stays linear in corpus size. `analyzer.extracted_unit_count` reports the pre-filter extraction count from the latest `index()` or `analyze()` run, which can be larger than the count returned by `index()` after semantic eligibility filtering. A search after `analyze()` reuses the analysis task and therefore requires an explicit search threshold when that task changes the model's prompt or route, as it does for EmbeddingGemma.

The contextual-threshold requirement follows the indexed representation even if the config changes afterward. Its [cache behavior](caching.md#what-invalidates-what) follows the complete document input. `analyze()` always embeds bare source for duplicate detection regardless of this search-only setting. The resolved `run.semantic.search_document` records which representation was actually embedded.

For direct embedding/query calls, pass the identity returned by `compute_embeddings_with_identity(...)` as `find_similar_to_query(corpus_identity=...)`. It is required for contextual documents and prompt- or route-sensitive models, and preserves calibration and checkpoint checks on both cold and warm cache paths. Use `search_document="contextual"` with aligned `document_texts` when supplying contextual inputs. The array-only `compute_embeddings(...)` helper accepts source documents only because it cannot return the identity needed to enforce contextual-search thresholds.

```python
from pathlib import Path

from codedupes.constants import DEFAULT_SEARCH_SEMANTIC_TASK
from codedupes.extractor import CodeExtractor
from codedupes.semantic import compute_embeddings_with_identity, find_similar_to_query

repo_root = Path("./src").resolve()
units = CodeExtractor(repo_root).extract_all()
embeddings, identity = compute_embeddings_with_identity(
    units,
    semantic_task=DEFAULT_SEARCH_SEMANTIC_TASK,
    cache_scope=repo_root,
)
hits = find_similar_to_query(
    "load csv data",
    units,
    embeddings,
    corpus_identity=identity,
    cache_scope=repo_root,
)
```

Direct `find_similar_to_query()` and `find_semantic_duplicates()` calls require a two-dimensional embedding matrix with exactly one row per supplied unit. Inputs are converted to float32 and unit-normalized before cosine comparison; non-finite or zero rows, and short, long, or one-dimensional matrices raise `ValueError` before cache or model work. A `(0, dimensions)` matrix remains valid for an empty unit list.

In `AnalyzerConfig(mode="search")`, `semantic_threshold` is any finite search floor, including a negative value when every ranked result is needed. Check-mode duplicate gates remain restricted to `[0, 1]`.

Direct `compute_embeddings()` and `compute_embeddings_with_identity()` calls require one `document_texts` entry per input unit when supplied. Mismatched lengths raise `ValueError` before revision resolution, cache lookup, or model loading, including for an empty corpus.

Each `index()` or `analyze()` call replaces the analyzer's corpus-specific state before extraction. `search()` therefore targets only the most recent run and requires it to have semantic embeddings. A later empty or nonsemantic analysis cannot reuse an older corpus accidentally. The analyzer binds the matrix to its model, revision, and vector-affecting runtime configuration. If any of those changes before a query, `search()` requires a fresh `index()`/`analyze()`. Strict [Hub revision resolution](caching.md#hub-revisions) is the default; set `AnalyzerConfig(strict_revision_cache=False)` only to opt into label-keyed warm hits.

## Progress and embedding telemetry

`AnalyzerConfig.progress` accepts `"auto"` (default), `"always"`, or `"never"`; other values raise `ValueError` at configuration construction. Auto mode renders embedding progress only for more than 100 uncached inputs when stderr is a TTY. The same keyword is available on `compute_embeddings`, `compute_embeddings_with_identity`, `run_semantic_analysis`, and `run_semantic_analysis_with_identity`.

The low-level corpus functions accept an `EmbeddingRunStats` collector through `stats=` and fill it in place. `find_similar_to_query()` likewise accepts an `execution=` list and appends a `QueryExecution` record after each query vector is successfully searched, including its effective encode device or whether the vector came from cache. `AnalysisResult.embedding_stats` contains corpus telemetry after successful semantic analysis; `CodeAnalyzer.embedding_stats` exposes it after `index()`, and `CodeAnalyzer.query_execution` retains the query records for that index. See [embedding telemetry](output.md#embedding-telemetry) for corpus counter definitions, cache warnings, and unavailable statistics.

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

Low-level `compute_embeddings*` calls require `cache_scope` for persistent reuse and do not publish [corpus manifests](caching.md#corpus-lifecycle); use `CodeAnalyzer` for move/deletion tracking.

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

See [Accelerators](accelerators.md) for fallback, OOM recovery, model placement, and the process-wide allocator policy. Long-lived processes can explicitly release the model:

```python
from codedupes.semantic import clear_model_cache

clear_model_cache()
```

## Logging

Python query calls log the effective search threshold at DEBUG. The human-readable CLI reports the resolved threshold during configuration; verbose DEBUG output also shows query-time resolution. [Family-threshold notices](model-profiles.md#alias-resolution-rules) are emitted once per model per process when automatic defaults are selected for a recognized copy or non-builtin Hub model.

Model loading quiets known-noisy dependency loggers (httpx request lines, transformers/sentence-transformers chatter) automatically, but only ones still inheriting the root level - any logger you configure explicitly is left alone. To pin them yourself, or to a different level:

```python
import logging

from codedupes import quiet_dependency_loggers

quiet_dependency_loggers()  # or quiet_dependency_loggers(logging.ERROR)
```

## Key result types

- `AnalysisResult.units`: extracted functions, methods, and classes
- `AnalysisResult.hybrid_duplicates`: every synthesized duplicate candidate with its [confidence tier](analysis-defaults.md#hybrid-synthesis-confidence-defaults); the CLI applies [report selection](#report-selection-and-json) on top of this complete list
- `AnalysisResult.traditional_duplicates`: raw traditional duplicates (diagnostics)
- `AnalysisResult.semantic_duplicates`: raw semantic duplicates (diagnostics)
- `AnalysisResult.potentially_unused`: Python-only [unused candidates](analysis-defaults.md#potentially-unused-defaults) from a name-based reference graph
- `AnalysisResult.extraction_diagnostics`: recoverable parser diagnostics and skipped-unit reasons
- `CodeAnalyzer.extraction_diagnostics`: extraction diagnostics from the latest `index()` or `analyze()` run
- `AnalysisResult.semantic_diagnostics`: semantic-stage diagnostics, mirroring `CodeAnalyzer.semantic_diagnostics` for that run
- `AnalysisResult.unused_diagnostics`: per-file diagnostics from the unused reference walk (`unused-read-error`, `unused-parse-error`, `unused-recursion-limit`); `codedupes.unused.run_unused_analysis` returns them, together with `unused` and `suppressed`, in an `UnusedReport`
- `AnalysisResult.unused_excluded_units`: non-Python units intentionally excluded from unused analysis
- `AnalysisResult.unused_supported_languages`: languages the unused heuristic evaluates (currently always `("python",)`)
- `AnalysisResult.suppressed_duplicates`: traditional and semantic pairs dropped for carrying a `codedupes: ignore[duplicates]` directive on either endpoint
- `AnalysisResult.suppressed_unused`: units carrying a `codedupes: ignore`/`codedupes: ignore[unused]` directive that would otherwise have been reported unused (same `suppressed` count `run_unused_analysis` returns on `UnusedReport`)
- `AnalysisResult.all_duplicates`: hybrid duplicates in combined mode; raw duplicates in single-method mode
- `AnalysisResult.analysis_mode`: derived from `run`; `"combined"` when both traditional and semantic ran, else `"traditional"`, `"semantic"`, or `"unused"`
- `AnalysisResult.run`: the resolved `RunRecord` this analysis actually applied — root/target, scope, per-detector settings (`traditional`, `semantic`, `unused`, each `None` when that detector did not run), and extraction/unit counts (`run.units` is a `UnitCounts`: `extracted`, `semantic_eligible`, and the `by_language`/`by_type` breakdowns, built by `UnitCounts.from_units(units, semantic_eligible=...)`); see [the run record](output.md#run-record-and-check-status)
- `AnalysisResult.checks`: `AnalysisChecks` derived from `run` and this result's diagnostics — one `CheckRecord(status, files, files_failed, diagnostics)` per detector, `status` one of `completed`, `partial`, `empty`, `fallback`, `disabled`
- `AnalysisResult.analysis_status`: `checks.analysis_status` — `"complete"`, `"partial"`, or `"empty"`
- `CodeAnalyzer.run_record`: the same `RunRecord` after the latest `index()` or `analyze()` call, or `None` before the first run; `index()` records semantic work even when a check-mode config has `run_semantic=False`, because that flag gates `analyze()`
- `AnalysisResult.embedding_stats`: [embedding telemetry](#progress-and-embedding-telemetry)
- `AnalysisResult.focus`: `FocusSummary(paths, units, out_of_focus_duplicates, out_of_focus_unused)` after `focus_result()`, else `None`; see [focused reports](output.md#focused-reports)
- `CodeUnit.uid`: in-run definition identity, `<path>::<language>::<qualified name>::<start byte>` for every language; the byte position keeps overloads and redefinitions distinct
- `CodeUnit.language`, `dialect`, and `native_kind`: canonical language, parser dialect, and grammar node kind (`function_definition`/`class_definition` for Python)
- `CodeUnit.start_byte`/`end_byte`: exact byte range used to slice the emitted source; a decorated Python definition starts at its first decorator
- `CodeUnit.structural_hash`, `token_hash`, `identifiers`, and `statement_count`: computed by the language backend from one Tree-sitter parse for every language; see [fingerprints](polyglot-languages.md#fingerprints-and-comparison-boundaries)
- `CodeUnit.suppressions`: [`codedupes: ignore`](analysis-defaults.md#suppression-directives) kinds attached to this unit, including any inherited from an enclosing unit's own directive; empty when none apply
- `HYBRID_TIERS`: the five tier names in declaration order, for zero-filled counts; pairs sort by [score](analysis-defaults.md#score-scale)

## Report selection and JSON

The CLI's report policy and [JSON schema](output.md#json-schema-v4) are importable, so Python callers can produce the same document as `check --json`:

```python
from codedupes import (
    DEFAULT_MAX_DUPLICATES,
    DEFAULT_MAX_UNUSED,
    ReportPolicy,
    check_result_to_json,
    run_should_fail,
    select_findings,
    to_json_text,
)

selection = select_findings(
    result, ReportPolicy(max_duplicates=DEFAULT_MAX_DUPLICATES, max_unused=DEFAULT_MAX_UNUSED)
)
exit_code = int(run_should_fail(result, policy="actionable", strict_unused=False))
print(
    to_json_text(
        check_result_to_json(
            selection, fail_on="actionable", exit_code=exit_code, strict_unused=False
        )
    )
)
```

`check_result_to_json(..., include_source=True, source_lines=40)` adds a bounded `source` and `source_lines_omitted` to every unit record, matching `--show-source`/`--source-lines`; both keyword arguments default to opting out, so the example above stays byte-identical to `check`'s default JSON.

`ReportPolicy()` is uncapped: `select_findings(result)` returns every default-visible finding, which is what `check --max-duplicates all --max-unused all` emits. The CLI's concise default is `ReportPolicy(max_duplicates=DEFAULT_MAX_DUPLICATES, max_unused=DEFAULT_MAX_UNUSED)` (20 and 20); `ReportPolicy(include_review=True)` and `ReportPolicy(show_all=True)` match `--include-review` and `--show-all`, which lift both caps on the command line. Either cap below `1` raises `ValueError`.

`select_findings` applies the visibility policy to a complete result and returns a `ReportSelection`. Exact edges of the primary list are grouped into `exact_families` (`ExactFamily` records with sorted `members`, a `method` of `token_hash` or `structural_hash`, and `lines`, `redundant_lines`, and `pair_count` properties; `build_exact_families(edges)` is the grouping itself), every other duplicate is a pair in `duplicates`, withheld pairs are `omitted_review`, and `truncated_exact_families` plus `truncated` hold what `max_duplicates` cut; `duplicates_by_tier` and `truncated_by_tier` are zero-filled with `exact` counting families. `potentially_unused` is ranked by `unused_sort_key` (line span, statement count, position) and capped by `max_unused`, with the cut units in `truncated_unused`; `units` holds the referenced units in report-id order. The finding counts JSON and the terminal print (`total_findings`, `reported_findings`, `truncated_findings`, `actionable_findings`, `reported_actionable_findings`, `exact_family_members`) are properties of the selection. In combined mode the primary list is ranked for review — families by `redundant_lines`, then actionable pair tiers, then `semantic_high_confidence`, then included `semantic_review` pairs, each pair group in the analyzer's score order — and the cap keeps a prefix of that ranking with a family counting once; `result.hybrid_duplicates` keeps the analyzer's order, exact pairs first. `actionable_pairs(pairs, combined=...)` is the shared pairwise filter behind the `actionable` policy. `run_should_fail` always evaluates the complete result, so hidden findings still count; `hidden_only_failure(selection, ...)` returns `{"review"}` when withheld `semantic_review` pairs are the only failing findings and an empty set otherwise (a cap cannot hide every failing finding because families and actionable tiers rank first). Both failure helpers raise `ValueError` for a `policy` outside `"actionable"`, `"all"`, and `"none"`. `run_should_fail(..., fail_on_incomplete=True)` also fails whenever `result.analysis_status != "complete"`, independent of `policy` (applies under `"none"` too); `hidden_only_failure` is unaffected by it. See [exit codes](output.md#exit-codes).

`focus_result(result, paths)` builds the result behind `check --focus`: pass a complete `AnalysisResult` and a tuple of validated, deduplicated paths (files or directories under the scan root — `codedupes.cli._options.resolve_focus_paths` performs that validation for the CLI and preserves an in-tree file symlink alias when its target is outside the root) to get back a new `AnalysisResult` with `traditional_duplicates`, `semantic_duplicates`, `hybrid_duplicates`, and `potentially_unused` scoped to those paths, `focus` set to a `FocusSummary`, and every other field — `units`, `run`, the diagnostics lists, `unused_excluded_units`, `embedding_stats` — left corpus-wide. An exact-duplicate family (from `build_exact_families`) is kept whole when any member is in focus, since consolidating it is one indivisible finding; every other duplicate pair is kept when either endpoint is in focus; a `potentially_unused` unit is kept when its file is in focus. `select_findings` and `run_should_fail` both work unmodified on the returned result, so `check`'s focused exit code and report are exactly `select_findings(focus_result(result, paths))` and `run_should_fail(focus_result(result, paths), ...)`. `FocusSummary.out_of_focus_duplicates`/`out_of_focus_unused` count what focusing removed, in the same finding units `total_findings` uses (a family counts once), so `select_findings(result).total_findings == select_findings(focus_result(result, paths)).total_findings + focus.out_of_focus_duplicates` always holds.

`search_result_to_json` serializes unit hits or [file search results](output.md#file-search). For file reports, fetch every matching unit before grouping so a file's contributors cannot exhaust the unit limit and hide other files:

```python
from codedupes import AnalyzerConfig, CodeAnalyzer, search_result_to_json, to_json_text
from codedupes.report import group_file_results

analyzer = CodeAnalyzer(AnalyzerConfig(mode="search", progress="never"))
indexed_units = analyzer.index("./src")
query = "load csv data"
hits = analyzer.search(query, top_k=max(indexed_units, 1))
payload = search_result_to_json(
    query,
    hits,
    indexed_units,
    analyzer.embedding_stats,
    run=analyzer.run_record,
    extraction_diagnostics=analyzer.extraction_diagnostics,
    semantic_diagnostics=analyzer.semantic_diagnostics,
    file_results=group_file_results(hits, top_k=5),
    query_execution=analyzer.query_execution,
)
print(to_json_text(payload))
```

For a unit report, set `search(query, top_k=...)` to the desired unit count and omit `file_results` from the serializer call. `run` is required: it is `analyzer.run_record` after `index()` (or `analyze()`) has populated it. `query_execution` is optional and defaults to `()`; pass `analyzer.query_execution` to include the latest query's threshold, cache, and device provenance in `summary.query_execution`, even after earlier searches on the same analyzer. The extraction check inside `run.checks` is derived from `run.units.extracted` (the pre-filter extraction count), not from `indexed_units` (the post-eligibility-filter search corpus size), so a corpus that extraction populated but semantic eligibility filtered down to zero reports `analysis_status: "complete"` rather than `"empty"`; see [the three empty cases](output.md#search).

## Notes

- `AnalyzerConfig` enforces workflow dependencies:
  - semantic-only settings require `run_semantic=True`, including `model_name`, `min_semantic_statements`, and `semantic_unit_types`
  - traditional-only settings require `run_traditional=True`
  - `strict_unused=True` requires `run_unused=True`
  - at least one of `run_traditional`, `run_semantic`, `run_unused` must be `True`; a config with every detector disabled raises `ValueError` at construction
- `device`, `mps_fallback`, and `mps_memory_fraction` require `run_semantic=True`. `embedding_cache=False` is accepted when semantic analysis is disabled and has no effect.
