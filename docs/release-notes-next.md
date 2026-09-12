# Next release

## Migration

- `--exclude` now extends default test exclusions for directory scans and matches directory descendants. Explicit file targets bypass default test patterns but honor custom exclusions. See [extraction scope](analysis-defaults.md#extraction-scope-defaults).
- JSON consumers must adopt [schema v3](output.md#json-schema-v3).
- `CodeUnit.uid` now includes language and start byte. The private `_ast_hash` alias, `has_body`, `AnalysisResult.filtered_raw_duplicates`, and the `AnalysisResult.exact_duplicates` alias (which returned every traditional pair, near-duplicates included, not the `exact` tier) were removed; read `traditional_duplicates` or filter `hybrid_duplicates` by `tier`. See [result types](python-api.md#key-result-types).
- `--min-lines` / `min_semantic_lines` became `--min-statements` / `min_semantic_statements`. The redundant `--tiny-near-jaccard-min` exception and `--hybrid-semantic-threshold` sweep flag were removed.
- Flat duplicate defaults were replaced by [per-language gates](analysis-defaults.md#semantic-duplicate-gate-defaults). Pass `--semantic-threshold` (or `AnalyzerConfig.semantic_threshold`) to retain a flat semantic policy.
- `codedupes check` now [withholds `semantic_review` pairs by default](output.md#report-selection). Consumers that require every hybrid edge should add `--include-review`.
- Default unused reporting now also [skips public methods of public classes](analysis-defaults.md#potentially-unused-defaults); pass `--strict-unused` to keep them. `build_reference_graph` and `find_potentially_unused` moved to `codedupes.unused`, and `run_traditional_analysis` returns `(exact, near)` only.
- Python is extracted with the exact-pinned `tree-sitter-python` grammar (a new runtime dependency) through the [same backend as every other language](polyglot-languages.md#python). A decorated definition is one unit from its first decorator, so `lineno`, `start_byte`, `start_column`, `source`, fingerprints, identifiers, and the `CodeUnit.uid` byte offset of decorated units move to the `@` line; `source` no longer ends with a newline and `start_column` is the real column.
- Python `native_kind` is `function_definition` or `class_definition` (was `FunctionDef`/`AsyncFunctionDef`/`ClassDef`), and a class defined inside a function is qualified `mod.func.Class` (was `mod.Class`).
- Python identifier sets now include attribute and keyword-argument names and exclude builtins, keywords, `self`, and `cls`; `codedupes.traditional.extract_identifiers` and `unit_identifier_set` were removed - read `CodeUnit.identifiers`. `CodeUnit.docstring` and `CodeUnit.calls` were removed, and `CodeUnit.statement_count` is always set at extraction.
- The `ast`-based extractor helpers `compute_ast_hash`, `compute_token_hash`, `count_executable_statements`, `get_exported_names`, `extract_docstring`, `NormalizedASTHasher`, `CallGraphVisitor`, and `_get_module_name` were removed with the Python `ast` extractor; the backend computes every feature. `codedupes.languages.registry.get_backend` no longer raises for Python. `semantic.get_code_unit_statement_count` no longer reparses a unit's source: it returns the extracted count, and a unit built without one measures 0. An indented unit's `source` (a method, a nested definition) no longer starts with its indentation.
- The exact-duplicate `method` label `ast_hash` is now `structural_hash` in `DuplicatePair.method`, the `--show-all` JSON edge lists, and the CLI table; it names the `CodeUnit.structural_hash` fingerprint every language shares.
- The Python-only `parse-error` diagnostic was replaced by the [shared codes](polyglot-languages.md#source-ranges-and-parse-recovery): a syntax error now yields `partial-parse` plus `unit-parse-error` for the broken unit while intact units are still extracted, and a non-UTF-8 file yields `invalid-utf8` and is analyzed after lossy decoding instead of being skipped.
- Search-only Python callers should use `AnalyzerConfig(mode="search")`; see the [search configuration](python-api.md#semantic-query-search). `analyze()` rejects that mode, while `index()` and `search()` support it.
- The default [Hub revision policy](caching.md#hub-revisions) now uses labels; `--strict-revision-cache` retains the previous policy.
- Runtime dependency minimums changed; use the [installation requirements](install.md). The C2LLM profile and DeepSpeed-only `gpu` extra were removed. Replace `semantic_profiles.resolve_model_name()` with `resolve_model_profile(...).canonical_name`.
- Source archives without VCS metadata build as `0.0.0+unknown`; tagged Git builds retain VCS-derived versions. Source distributions use an explicit file allowlist.

## Detection and extraction

- Added [C, Rust, JavaScript/JSX, and TypeScript/TSX extraction](polyglot-languages.md), and moved Python onto the same Tree-sitter path so all five languages share one fingerprint, identifier, statement-count, and diagnostic implementation. Unused analysis remains Python-only.
- Unused analysis now builds its [reference graph](analysis-defaults.md#potentially-unused-defaults) from every loaded name, attribute access, annotation, import, and module-level statement rather than call sites alone, and treats public methods of framework-derived classes (`ast.NodeVisitor`, `logging.Filter`) as referenced, resolving their bases through the module's import and assignment aliases. Every Python file the extractor visits is parsed for references, units or not (re-export modules, scripts), and `CodeExtractor.extracted_files` records the visited files per language. `[project.entry-points]` groups seed entry points alongside `scripts`/`gui-scripts`. A unit's own body never counts for itself, methods included; `Literal[...]` values are not names; the `@abstractmethod` exemption reads only the unit's own decorators; and default mode reports a public definition nested in a private function or class. A file the standard-library parser rejects, or that overflows its recursion limit, logs a warning and contributes no references instead of aborting the run.
- Python `;` statement separators, optional trailing commas (magic trailing commas included), grouping parentheses, and plain implicit string concatenations are [formatting](polyglot-languages.md#fingerprints-and-comparison-boundaries): they no longer move the structural hash, while the token hash still sees them. Tuple-forming subscript commas remain structural, so `data[key]` and `data[key,]` do not match exactly. A `("doc")` docstring prunes like a bare one; a concatenation with an f-string part is still walked. C, Rust, JavaScript, and TypeScript fingerprints are unchanged.
- Python `__all__` accepts bare tuples and module-level `if`/`try` bodies, a definition with no body yields no unit, a filtered private definition of any kind drops its nested definitions, and identifier sets keep the `site`-injected names (`exit`, `quit`, `help`) that are not language builtins.
- Improved language-specific extraction, traditional matching, and source-range handling; see [polyglot language support](polyglot-languages.md).
- Added [polyglot calibration corpora](../test_fixtures/polyglot_calibration/README.md) and a runnable [Rust/WebAssembly clone fixture](../test_fixtures/cowsay_wasm/README.md).
- The [hybrid confidence split](analysis-defaults.md#hybrid-synthesis-confidence-defaults) is calibrated per model profile. [Recorded calibration results](../test_fixtures/polyglot_calibration/README.md#calibration-results) support the shipped values.

## Semantic inference and caching

- Recognized local copies and fine-tunes retain their family's tuned thresholds; use `--threshold-profile generic` to restore generic defaults. See [model profiles](model-profiles.md).
- Added persistent [embedding caching](caching.md) and [corpus lifecycle tracking](caching.md#corpus-lifecycle).
- Added explicit CPU/CUDA/MPS selection, dtype control, allocator diagnostics, and bounded OOM recovery. See [accelerator behavior](accelerators.md).
- Added [search indexing, per-query thresholds, and contextual documents](python-api.md#semantic-query-search).
- Long definitions and queries now use normal embedding-backend truncation rather than being excluded from semantic analysis.
- Improved model fingerprints, cache recovery, semantic-pair scanning, and traditional Jaccard matching.

## CLI and API output

- Added `search --result-level file` to rank distinct files by their strongest matching code unit, with brief contributing-unit details in terminal and JSON reports. Unit-level search remains the default.
- Added grouped help, `-h`, paired boolean flags, and command-line-only CLI configuration: `CODEDUPES_*` environment variables no longer override options. See [CLI options](cli.md).
- Added configurable [finding exit policies](output.md#exit-codes), [embedding telemetry](output.md#embedding-telemetry), and clean JSON output under merged streams.
- Report selection and JSON serialization are exported from `codedupes`; `run_should_fail` now uses `AnalysisResult.analysis_mode` instead of a caller-supplied `combined_mode` flag. See [report selection](python-api.md#report-selection-and-json).
- Added `check --max-duplicates N`; see [report selection](output.md#report-selection).
- Improved paths, search queries, empty-index reporting, reused analyzers, cache failures, and command validation.
