# Next release

## Migration

- `--exclude` now extends default test exclusions for directory scans and matches directory descendants. Explicit file targets bypass default test patterns but honor custom exclusions. See [extraction scope](analysis-defaults.md#extraction-scope-defaults).
- JSON consumers must adopt [schema v3](output.md#json-schema-v3).
- `CodeUnit.uid` now includes language and start byte. The private `_ast_hash` alias, `has_body`, `AnalysisResult.filtered_raw_duplicates`, and the `AnalysisResult.exact_duplicates` alias (which returned every traditional pair, near-duplicates included, not the `exact` tier) were removed; read `traditional_duplicates` or filter `hybrid_duplicates` by `tier`. See [result types](python-api.md#key-result-types).
- `--min-lines` / `min_semantic_lines` became `--min-statements` / `min_semantic_statements`. The redundant `--tiny-near-jaccard-min` exception and `--hybrid-semantic-threshold` sweep flag were removed.
- Flat duplicate defaults were replaced by [per-language gates](analysis-defaults.md#semantic-duplicate-gate-defaults). Pass `--semantic-threshold` (or `AnalyzerConfig.semantic_threshold`) to retain a flat semantic policy.
- `codedupes check` now [withholds `semantic_review` pairs by default](output.md#report-selection). Consumers that require every hybrid edge should add `--include-review`.
- The default `check` report, terminal and JSON alike, now lists at most 20 primary duplicate pairs with actionable tiers first; `--json` used to emit every default-visible pair while the terminal showed 20. Add `--max-duplicates all` to restore the complete default-visible list, or `--show-all` for every hybrid tier plus the raw lists; `--include-review`, `--show-all`, and `--full-table` lift the cap unless an explicit `--max-duplicates` is given. `check_result_to_json` now requires `strict_unused`. See [report selection](output.md#report-selection).
- Default unused reporting now also [skips public methods of public classes](analysis-defaults.md#potentially-unused-defaults); pass `--strict-unused` to keep them. `build_reference_graph` and `find_potentially_unused` moved to `codedupes.unused`, and `run_traditional_analysis` returns `(exact, near)` only.
- Python is extracted with the exact-pinned `tree-sitter-python` grammar (a new runtime dependency) through the [same backend as every other language](polyglot-languages.md#python). A decorated definition is one unit from its first decorator, so `lineno`, `start_byte`, `start_column`, `source`, fingerprints, identifiers, and the `CodeUnit.uid` byte offset of decorated units move to the `@` line; `source` no longer ends with a newline and `start_column` is the real column.
- Python `native_kind` is `function_definition` or `class_definition` (was `FunctionDef`/`AsyncFunctionDef`/`ClassDef`), and a class defined inside a function is qualified `mod.func.Class` (was `mod.Class`).
- Python identifiers now include attributes and keyword arguments while excluding builtins, keywords, `self`, and `cls`; use `CodeUnit.identifiers` instead of the removed identifier helpers. `CodeUnit.docstring` and `CodeUnit.calls` were removed, while `statement_count` is always extracted.
- The Python `ast` helpers (`compute_ast_hash`, `compute_token_hash`, `count_executable_statements`, `get_exported_names`, `extract_docstring`, `NormalizedASTHasher`, `CallGraphVisitor`, and `_get_module_name`) were removed with that backend; Tree-sitter supplies those features. `get_code_unit_statement_count` returns the extracted count (or `0` for a hand-built unit), and indented unit sources are dedented.
- The exact-duplicate `method` label `ast_hash` is now `structural_hash` in `DuplicatePair.method`, the `--show-all` JSON edge lists, and the CLI table; it names the `CodeUnit.structural_hash` fingerprint every language shares.
- The Python-only `parse-error` diagnostic was replaced by the [shared codes](polyglot-languages.md#source-ranges-and-parse-recovery): a syntax error now yields `partial-parse` plus `unit-parse-error` for the broken unit while intact units are still extracted, and a non-UTF-8 file yields `invalid-utf8` and is analyzed after lossy decoding instead of being skipped.
- Search-only Python callers should use `AnalyzerConfig(mode="search")`; see the [search configuration](python-api.md#semantic-query-search). `analyze()` rejects that mode, while `index()` and `search()` support it.
- Unpinned Hub models use concrete locally resolved commits for cache identity by default. `--loose-revision-cache` opts into label-keyed warm hits that may remain stale after an upstream branch move.
- Runtime dependency minimums changed; use the [installation requirements](install.md). The C2LLM profile and DeepSpeed-only `gpu` extra were removed. Replace `semantic_profiles.resolve_model_name()` with `resolve_model_profile(...).canonical_name`.
- Source archives without VCS metadata build as `0.0.0+unknown`; tagged Git builds retain VCS-derived versions. Source distributions use an explicit file allowlist.

## Detection and extraction

- Added [C, Rust, JavaScript/JSX, and TypeScript/TSX extraction](polyglot-languages.md), and moved Python onto the same Tree-sitter path so all five languages share one fingerprint, identifier, statement-count, and diagnostic implementation. Function and method fingerprints now share a comparison domain; unused analysis remains Python-only.
- Unused analysis now uses a broader [reference graph](analysis-defaults.md#potentially-unused-defaults), including imports, annotations, attributes, and module statements. It analyses every visited Python file, honors package entry points and framework methods, and degrades malformed files to warnings rather than aborting a scan.
- Python `;` statement separators, optional trailing commas (magic trailing commas included), grouping parentheses, and plain implicit string concatenations are [formatting](polyglot-languages.md#fingerprints-and-comparison-boundaries): they no longer move the structural hash, while the token hash still sees them. Tuple-forming subscript commas remain structural, so `data[key]` and `data[key,]` do not match exactly. A `("doc")` docstring prunes like a bare one; a concatenation with an f-string part is still walked. Formatting policies for C, Rust, JavaScript, and TypeScript are unchanged.
- Python `__all__` accepts bare tuples and module-level `if`/`try` bodies, a definition with no body yields no unit, a filtered private definition of any kind drops its nested definitions, and identifier sets keep the `site`-injected names (`exit`, `quit`, `help`) that are not language builtins.
- Replaced the previous calibration data and compatibility paths with a [versioned manifest and explicit pair contract](../test_fixtures/calibration/README.md) covering runnable Python, C, Rust, JavaScript, and TypeScript applications. Positive pairs span easy, medium, and hard rewrites; nearby negatives and search relevance are reviewed independently.
- Recalibrated both built-in profiles from uncached CPU fp32 measurements, then verified independent MPS fp32 runs. Per-language duplicate gates, search gates, and hybrid visibility constants now use the [checked development-corpus result](../test_fixtures/calibration/calibration-results.json); the [workflow and phase scope](hybrid-tuning.md) explain this first Issue #20 phase, while raw score matrices remain local scratch data.

## Semantic inference and caching

- Recognized local copies and fine-tunes retain their family's tuned thresholds; use `--threshold-profile generic` to restore generic defaults. See [model profiles](model-profiles.md).
- Added persistent [embedding caching](caching.md) and [corpus lifecycle tracking](caching.md#corpus-lifecycle).
- Added explicit CPU/CUDA/MPS selection, dtype control, allocator diagnostics, and bounded OOM recovery. See [accelerator behavior](accelerators.md).
- Added [search indexing, per-query thresholds, and contextual documents](python-api.md#semantic-query-search).
- Long definitions and queries now use normal embedding-backend truncation rather than being excluded from semantic analysis.

## CLI and API output

- Added `search --result-level file` to rank distinct files by their strongest matching code unit, with brief contributing-unit details in terminal and JSON reports. Unit-level search remains the default.
- Added grouped help, `-h`, paired boolean flags, and command-line-only CLI configuration: `CODEDUPES_*` environment variables no longer override options. See [CLI options](cli.md).
- Added configurable [finding exit policies](output.md#exit-codes), [embedding telemetry](output.md#embedding-telemetry), and clean JSON output under merged streams.
- Report selection and JSON serialization are exported from `codedupes`; `run_should_fail` now uses `AnalysisResult.analysis_mode` instead of a caller-supplied `combined_mode` flag. See [report selection](python-api.md#report-selection-and-json).
- Added `check --max-duplicates <N|all>` (default `20`) and the `actionable_duplicates`, `reported_actionable_duplicates`, `strict_unused`, and `hidden_only_failure` summary fields; the terminal summary gained an `Actionable duplicates` row. See [report selection](output.md#report-selection).
