# Next release

This draft changelog is for people upgrading an existing integration. Read the migration section before updating CI commands or Python callers; new users should start with the [README](../README.md) and [installation guide](install.md).

## Migration

- CLI `--exclude` now extends default test exclusions for directory scans and matches directory descendants. Explicit file targets bypass default test patterns but honor custom exclusions. Use `--no-default-excludes` to include tests in directory scans; Python `exclude_patterns=[]` now disables test defaults. See [extraction scope](analysis-defaults.md#extraction-scope-defaults).
- JSON consumers must adopt [schema v3](output.md#json-schema-v3): findings reference report-local unit ids (`u0`, `u1`, ...) in a top-level `units` map whose records carry the long `uid`, instead of full unit objects at each pair endpoint, and `summary` gains `reported_duplicates`, `omitted_review_duplicates`, `truncated_duplicates`, `max_duplicates`, and `duplicates_by_tier`.
- `CodeUnit.uid` now includes language and start byte. The private `_ast_hash` alias, `has_body`, and `AnalysisResult.filtered_raw_duplicates` were removed. See [result types](python-api.md#key-result-types).
- `--min-lines` / `min_semantic_lines` became `--min-statements` / `min_semantic_statements`. The redundant `--tiny-near-jaccard-min` exception and `--hybrid-semantic-threshold` sweep flag were removed.
- Flat duplicate defaults were replaced by [per-language gates](analysis-defaults.md#semantic-duplicate-gate-defaults). Pass `--semantic-threshold` (or `AnalyzerConfig.semantic_threshold`) to retain a flat semantic policy without also changing the traditional threshold. Semantic-only matches are no longer dropped below a second synthesis threshold; they are synthesized and counted, and the weakest tier is withheld from the default report (next bullet).
- `codedupes check` now [withholds `semantic_review` pairs by default](output.md#json-schema-v3) in both terminal and JSON output; pass `--include-review` to list them (`--show-all` implies it). Exit codes are still computed on the complete result, so `--fail-on all` continues to fail on withheld pairs and the terminal says so when they are the only failing finding. Consumers that scripted over every hybrid edge should add `--include-review`. On this repository's `src/` (no exact or Jaccard pairs) the previous defaults reported all 519 hybrid pairs, 370 of them `semantic_review`; the default report now lists 114 `semantic_high_confidence` pairs and withholds 401 under the re-tuned gte split, with `summary.omitted_review_duplicates` carrying the count.
- Search-only Python callers should use `AnalyzerConfig(mode="search")`; see the [search configuration](python-api.md#semantic-query-search). `analyze()` rejects that mode, while `index()` and `search()` support it.
- The default [Hub revision policy](caching.md#hub-revisions) now uses labels; `--strict-revision-cache` retains the previous policy.
- Runtime dependency minimums changed; use the [installation requirements](install.md). The C2LLM profile and DeepSpeed-only `gpu` extra were removed. Replace `semantic_profiles.resolve_model_name()` with `resolve_model_profile(...).canonical_name`.
- Source archives without VCS metadata build as `0.0.0+unknown`; tagged Git builds retain VCS-derived versions. Source distributions use an explicit file allowlist.

## Detection and extraction

- Added [C, Rust, JavaScript/JSX, and TypeScript/TSX extraction](polyglot-languages.md), language filters, byte ranges, parser readiness, and recoverable diagnostics. Unused analysis remains Python-only.
- Fixed language-specific visibility, export scope, trait-qualified names, bound class identities, Unicode identifiers, and structural normalization. Grammar fixtures now check extraction and fingerprint behavior across upgrades.
- Traditional matching now keeps full extraction scope independently of semantic candidate filters. Tiny-class filtering uses member statement counts. Functions and methods share a comparison kind.
- Python byte ranges preserve BOM/CRLF offsets, unreadable files emit diagnostics, and sorted walks keep result order stable. Exact token hashes and embedding inputs normalize line endings.
- Exact pairs suppressed by the tiny filter cannot reappear as semantic findings. Python identifier normalization now uses the actual built-in name set.
- Added [polyglot calibration corpora](../test_fixtures/polyglot_calibration/README.md) and a runnable [Rust/WebAssembly clone fixture](../test_fixtures/cowsay_wasm/README.md).
- The `semantic_high_confidence` / `semantic_review` split is now [calibrated per model profile](analysis-defaults.md#hybrid-synthesis-confidence-defaults) instead of fixed at a `0.20` identifier Jaccard and `0.35` statement ratio: `gte-modernbert-base` promotes on a `0.80` statement ratio or a per-language similarity gate (TypeScript `0.88`), `embeddinggemma-300m` on a `0.20` ratio, and neither requires identifier overlap because the Python extractor collects no attribute names. On the polyglot corpus gte's high-confidence subset measures precision 0.78 / recall 0.57 against 0.68 / 0.62 for everything admitted; gemma's corpus rows are unchanged. `scripts/sweep_hybrid_gates.py` now sweeps both profiles across every language and records `test_fixtures/polyglot_calibration/reports/corroboration_report.json`, which `tests/test_corroboration_reports.py` ties to the shipped values. The five per-language semantic threshold reports were regenerated with per-tier (`tiers`) and default-visible (`visible`) metrics on every row plus the `corroboration` split they were scored with; `tests/test_calibration_reports.py` now also checks that the recorded split matches the profile and that withholding `semantic_review` never lowers precision at a shipped gate.

## Semantic inference and caching

- Recognized local copies and fine-tunes now retain their family's tuned thresholds. Configuration-based recognition takes precedence over directory names; local official EmbeddingGemma copies need no README or online checkpoint verification. Use `--threshold-profile generic` to restore generic defaults, or select a named profile through the CLI/Python API. Numeric overrides and custom-context requirements remain in effect.
- Local model fingerprints now ignore documentation and Git/download metadata while continuing to track weights, tokenizer assets, configuration, pooling/Dense modules, and custom code. Existing local caches may miss once; changing a threshold profile reuses embeddings.
- Added persistent [embedding caching](caching.md) and [corpus lifecycle tracking](caching.md#corpus-lifecycle).
- Added explicit CPU/CUDA/MPS selection, dtype control, allocator diagnostics, and bounded OOM recovery. See [accelerator behavior](accelerators.md).
- Fixed task prompts being applied twice and added [model-context calibration requirements](model-profiles.md#semantic-task-defaults-and-choices).
- Added [linear-time search indexing, per-query thresholds, and contextual documents](python-api.md#semantic-query-search).
- Eligible long definitions and queries now use normal embedding-backend truncation rather than being excluded from semantic analysis. Newly encoded over-context units emit warning diagnostics without losing their embedding rows; cache hits do not repeat those warnings.
- Local-model fingerprints, revision provenance, and runtime identities prevent mixing vectors from different model states. Corrupt cache rows become misses and repair on recomputation.
- Semantic pair scanning now thresholds NumPy row-block products; traditional Jaccard matching uses a prefix-filtered join. Recorded 8,000-unit comparisons improved from 3.5 s to 0.08 s and 55.6 s to 0.42 s respectively, with equivalence tests for pairs, scores, and order. Rust attribute traversal also avoids repeated linear sibling scans.

## CLI and API output

- Added `search --result-level file` to rank distinct files by their strongest matching code unit, with brief contributing-unit details in terminal and JSON reports. Unit-level search remains the default.
- Split the CLI into command, option, and rendering modules. Added grouped help, `-h`, and paired boolean flags. CLI options use command-line flags without automatic environment-variable overrides. See [CLI options](cli.md).
- Added configurable [finding exit policies](output.md#exit-codes), [embedding telemetry](output.md#embedding-telemetry), and clean JSON output under merged streams.
- Report selection and JSON serialization moved into `codedupes.report` and are exported from `codedupes` (`select_findings`, `ReportPolicy`, `check_result_to_json`, `search_result_to_json`, `run_should_fail`, `hidden_only_failure`); see [report selection](python-api.md#report-selection-and-json). `run_should_fail` now keys on `AnalysisResult.analysis_mode` instead of a caller-supplied `combined_mode` flag.
- Added `check --max-duplicates N` (`ReportPolicy(max_duplicates=N)`) to bound the number of emitted duplicate pairs in JSON and terminal output; it keeps the highest-confidence prefix after the review filter, records `summary.truncated_duplicates` and `summary.max_duplicates`, leaves the `--show-all` raw lists whole, and never changes the exit code. When a run fails only over hidden pairs, the terminal status names the group (withheld review or truncated) and the flag that lists it.
- Paths retain enough context to distinguish files in different directories, including literal Rich markup characters. Search queries also preserve literal markup syntax in terminal output. Empty search indexes distinguish extraction and eligibility filtering.
- Reused analyzers clear prior corpus state. Python callers can control [progress](python-api.md#progress-and-embedding-telemetry) and [dependency logging](python-api.md#logging).
- Cache deletion failures now return a failing status, unavailable explicit accelerators are validated on warm and empty runs, and contradictory command options fail validation.
- Empty `cache clear --model` scopes are rejected without deleting entries. Contextual search requires a threshold before indexing; search construction and missing model-file failures use the normal stderr error path.

## Validation

Filesystem transition tests cover cold/warm scans, edits, moves, deletes, shared inputs, narrow runs, and cached/uncached result parity. [Hardware suites](accelerators.md#hardware-validation) exercise real CUDA and MPS devices and skip when unavailable.

Recorded release validation on an Apple M5 (32 GB, macOS Tahoe, PyTorch 2.13.0) completed the MPS suite and strict unsupported-op runs. A full check of this repository was roughly 38 times faster than CPU and returned the same pairs; raw similarities differed by about 0.0002. These measurements describe that run, not a performance guarantee.
