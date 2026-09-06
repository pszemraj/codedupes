# Next release

## Migration

- CLI `--exclude` now extends default test exclusions and matches directory descendants. Use `--no-default-excludes` to scan tests; Python `exclude_patterns=[]` now disables test defaults. See [extraction scope](analysis-defaults.md#extraction-scope-defaults).
- JSON consumers must adopt [schema v2](output.md#json-schema-v2) instead of expecting full unit objects at each pair endpoint.
- `CodeUnit.uid` now includes language and start byte. The private `_ast_hash` alias, `has_body`, and `AnalysisResult.filtered_raw_duplicates` were removed. See [result types](python-api.md#key-result-types).
- `--min-lines` / `min_semantic_lines` became `--min-statements` / `min_semantic_statements`. The redundant `--tiny-near-jaccard-min` exception and `--hybrid-semantic-threshold` sweep flag were removed.
- Flat duplicate defaults were replaced by [per-language gates](analysis-defaults.md#semantic-duplicate-gate-defaults). Pass an explicit threshold to retain a flat policy. Semantic-only matches now remain visible as review candidates rather than disappearing below a second synthesis threshold.
- Search-only Python callers should use the [search configuration](python-api.md#semantic-query-search).
- Embedding pipeline schema 5 invalidates older cache entries that could contain prefix-truncated vectors. The first run re-embeds them under the [complete-definition context policy](analysis-defaults.md#semantic-candidate-defaults).
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

## Semantic inference and caching

- Added persistent [embedding caching](caching.md) and [corpus lifecycle tracking](caching.md#corpus-lifecycle).
- Added explicit CPU/CUDA/MPS selection, dtype control, allocator diagnostics, and bounded OOM recovery. See [accelerator behavior](accelerators.md).
- Fixed task prompts being applied twice and added [model-context calibration requirements](model-profiles.md#semantic-task-defaults-and-choices).
- Added [linear-time search indexing, per-query thresholds, and contextual documents](python-api.md#semantic-query-search).
- Local-model fingerprints, revision provenance, and runtime identities prevent mixing vectors from different model states. Corrupt cache rows become misses and repair on recomputation.
- Semantic pair scanning now thresholds NumPy row-block products; traditional Jaccard matching uses a prefix-filtered join. Recorded 8,000-unit comparisons improved from 3.5 s to 0.08 s and 55.6 s to 0.42 s respectively, with equivalence tests for pairs, scores, and order. Rust attribute traversal also avoids repeated linear sibling scans.

## CLI and API output

- Split the CLI into command, option, and rendering modules. Added grouped help, `-h`, and paired boolean flags. CLI options use command-line flags without automatic environment-variable overrides. See [CLI options](cli.md).
- Added configurable [finding exit policies](output.md#exit-codes), [embedding telemetry](output.md#embedding-telemetry), and clean JSON output under merged streams.
- Paths retain enough context to distinguish files in different directories, including literal Rich markup characters. Empty search indexes distinguish extraction, eligibility, and context-window exclusions.
- Reused analyzers clear prior corpus state. Python callers can control [progress](python-api.md#progress-and-embedding-telemetry) and [dependency logging](python-api.md#logging).
- Cache deletion failures now return a failing status, unavailable explicit accelerators are validated on warm and empty runs, and contradictory command options fail validation.
- Empty `cache clear --model` scopes are rejected without deleting entries. Contextual search requires a threshold before indexing; search construction and missing model-file failures use the normal stderr error path.

## Validation

Filesystem transition tests cover cold/warm scans, edits, moves, deletes, shared inputs, narrow runs, and cached/uncached result parity. [Hardware suites](accelerators.md#hardware-validation) exercise real CUDA and MPS devices and skip when unavailable.

Recorded release validation on an Apple M5 (32 GB, macOS Tahoe, PyTorch 2.13.0) completed the MPS suite and strict unsupported-op runs. A full check of this repository was roughly 38 times faster than CPU and returned the same pairs; raw similarities differed by about 0.0002. These measurements describe that run, not a performance guarantee.
