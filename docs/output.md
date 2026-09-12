# Output and exit codes

## Report streams

For `check` and `search`, stdout contains the report: JSON under `--json` and Rich
tables otherwise. Errors and parser-unavailable remediation use stderr; Rich mode also
sends logs, cache warnings, sentence-transformers progress, and Hugging Face download
progress there. A completed JSON report is a single parseable JSON document even when
`check` exits `1` for findings. JSON mode disables progress and records non-fatal cache
failures in `summary.embeddings.cache_warnings` instead of emitting them. Runtime
failures restore stderr and do not produce a completed JSON report.

Terminal reports fit the available width. Below 120 columns, duplicate tables stack
their metrics and both code locations into **Evidence** and **Code units** columns.
Search scores keep their own columns, and long names and paths wrap within rows.

Write a JSON report directly in automation:

```text
codedupes check ./src --json > codedupes-report.json
```

On a completed scan, the report is written before the command returns its [finding
status](#exit-codes). Use `--fail-on none` to collect a report without making findings
fail an incremental rollout:

```text
codedupes check ./src --json --fail-on none > codedupes-report.json
```

If a Bash or Zsh pipeline validates JSON with `jq`, enable `pipefail` so the parser's successful
exit does not hide `codedupes`' status:

```text
set -o pipefail
codedupes check ./src --json | jq empty
```

## JSON schema v3

`check --json` and `search --json` emit schema version `3`. Units are nodes in a top-level `units` object keyed by report-local ids (`u0`, `u1`, ...); findings refer to those ids instead of repeating a complete unit object for every pair endpoint. Ids are assigned in file-path then source-offset order over the referenced units only, so they renumber whenever the referenced set changes (for example with `--include-review`) — treat them as opaque within one report. Each unit record also carries the [in-run `CodeUnit.uid`](python-api.md#key-result-types), not a cross-machine finding identifier.

### Check

```json
{
  "schema_version": 3,
  "analysis_mode": "combined",
  "summary": {
    "total_units": 42,
    "units_by_language": {"python": 42},
    "hybrid_duplicates": 3,
    "reported_duplicates": 1,
    "omitted_review_duplicates": 2,
    "truncated_duplicates": 0,
    "max_duplicates": null,
    "duplicates_by_tier": {
      "exact": 0,
      "traditional_near": 0,
      "hybrid_confirmed": 1,
      "semantic_high_confidence": 0,
      "semantic_review": 2
    },
    "potentially_unused": 1,
    "raw_traditional_duplicates": 1,
    "raw_semantic_duplicates": 3,
    "semantic_fallback": false,
    "semantic_fallback_reason": null,
    "extraction_diagnostics": 0,
    "semantic_diagnostics": 0,
    "unused_supported_languages": ["python"],
    "unused_excluded_units": 0,
    "embeddings": {
      "requested_rows": 40,
      "unique_inputs": 39,
      "cache_hit_rows": 38,
      "duplicate_rows_reused": 1,
      "encoded_inputs": 1,
      "model_loaded": true,
      "cache_enabled": true,
      "cache_warnings": [],
      "cache_revision": "0123456789abcdef",
      "execution_device": "cuda:0",
      "moved_units_reused": 0,
      "deleted_units": 0,
      "orphan_rows_retained": 2,
      "orphan_rows_collected": 0,
      "manifest_generation": 17
    },
    "fail_on": "actionable",
    "exit_code": 1
  },
  "duplicates": [
    {
      "unit_a": "u0",
      "unit_b": "u1",
      "tier": "hybrid_confirmed",
      "confidence": 0.94,
      "has_exact": false,
      "semantic_similarity": 0.96,
      "jaccard_similarity": 0.92,
      "weak_identifier_jaccard": 0.7,
      "statement_count_ratio": 1.0
    }
  ],
  "potentially_unused": ["u2"],
  "extraction_diagnostics": [],
  "semantic_diagnostics": [],
  "units": {
    "u0": {
      "uid": "/repo/src/a.py::python::a.normalize::0",
      "name": "normalize",
      "qualified_name": "a.normalize",
      "type": "function",
      "language": "python",
      "dialect": "python",
      "native_kind": "FunctionDef",
      "file": "/repo/src/a.py",
      "line": 1,
      "end_line": 2,
      "start_byte": 0,
      "end_byte": 42,
      "start_column": 0,
      "end_column": 24,
      "statement_count": 1,
      "is_public": true,
      "is_exported": false
    }
  }
}
```

The shortened example omits `u1` and `u2` from `units`; real output includes every id referenced by any emitted finding list exactly once. Units with no emitted finding are not present, so `summary.total_units` is the full extracted corpus count while `units` contains only units needed to resolve the report.

#### Report selection

In default combined mode, `duplicates` contains hybrid edges of every tier except `semantic_review`; see [tier evidence and confidence](analysis-defaults.md#hybrid-synthesis-confidence-defaults). `--include-review` emits the review tier too (ordered after the reported tiers, as the analyzer ranks them). `--show-all` implies `--include-review` and also adds `traditional_duplicates` and `semantic_duplicates` as raw edge lists with `unit_a`, `unit_b`, `similarity`, and `method`.

`summary.hybrid_duplicates` counts the complete synthesis, `summary.duplicates_by_tier` breaks that count down over all five tiers (always present, zero-filled), `summary.reported_duplicates` counts the edges actually emitted, `summary.omitted_review_duplicates` counts pairs withheld by the report policy, and `summary.truncated_duplicates` counts pairs cut by `--max-duplicates`, so `reported_duplicates + omitted_review_duplicates + truncated_duplicates == hybrid_duplicates` regardless of flags.

Nothing is truncated unless you ask: `--max-duplicates N` keeps the first `N` edges of the admitted list in the analyzer's confidence order (so the strongest evidence survives), records the cap as `summary.max_duplicates` (`null` when unset), and drops units referenced only by cut edges from `units`. The cap applies after the review filter, so `--include-review --max-duplicates N` ranks review pairs into the same budget; the raw `--show-all` lists are never capped. The exit code ignores the cap, see [exit codes](#exit-codes).

In `--semantic-only` or `--traditional-only` mode, `duplicates` directly contains the active raw edge list ordered by descending similarity (exact pairs at 1.0 first, ties in analyzer order; `--max-duplicates` keeps that prefix), `duplicates_by_tier` is all zeros, and the `--show-all` arrays are omitted. `analysis_mode` is always one of `combined`, `traditional`, `semantic`, or `none`.

See [hybrid confidence tiers](analysis-defaults.md#hybrid-synthesis-confidence-defaults) to interpret `tier` and `confidence`.

### Search

Default search hits (`--result-level unit`) use `{"unit": "u0", "score": 0.95}`; their unit records have the same fields as check results. An empty index with `--no-cache` produces:

```json
{
  "schema_version": 3,
  "query": "refund validation",
  "summary": {
    "indexed_units": 0,
    "results": 0,
    "embeddings": {
      "requested_rows": 0,
      "unique_inputs": 0,
      "cache_hit_rows": 0,
      "duplicate_rows_reused": 0,
      "encoded_inputs": 0,
      "model_loaded": false,
      "cache_enabled": false,
      "cache_warnings": [],
      "cache_revision": null,
      "execution_device": null,
      "moved_units_reused": 0,
      "deleted_units": 0,
      "orphan_rows_retained": 0,
      "orphan_rows_collected": 0,
      "manifest_generation": null
    }
  },
  "results": [],
  "units": {},
  "extraction_diagnostics": [],
  "semantic_diagnostics": []
}
```

`summary.indexed_units` is the semantic corpus size after eligibility filtering. An empty terminal index warns on stderr and distinguishes empty extraction from eligibility filtering.

#### File search

`search --result-level file --json` adds top-level `"result_level": "file"` and returns one `results` entry per matching file:

```json
{
  "file": "/repo/src/parser.py",
  "score": 0.91,
  "matching_units": 2,
  "matches": [
    {"unit": "u0", "score": 0.91},
    {"unit": "u1", "score": 0.86}
  ]
}
```

The file's `score` is its highest unit score. `matching_units` counts all of that file's units above the search threshold; `matches` contains up to three strongest contributors. Their ids reference the top-level `units` map, whose records carry `uid`, names, types, and line ranges. Only these displayed contributors appear in `units`. Files are ranked before applying `--top-k`; `summary.results` counts returned files, while `summary.indexed_units` still counts indexed code units. Diagnostics and embedding telemetry keep the same shape. Unit-level output remains the default and does not add `result_level`.

## Embedding telemetry

`summary.embeddings` describes the final retained corpus from the most recent embedding call; it is `null` when semantic work did not run or fell back. Query encoding does not increment these counters, though query-cache warnings are appended. A warm corpus can therefore report `model_loaded: false` while a new query still loads the model.

| Field | Meaning |
| --- | --- |
| `requested_rows` | Corpus rows selected for embedding. |
| `unique_inputs` | Distinct prepared texts among those rows. |
| `cache_hit_rows` | Corpus rows supplied by persistent cache, including repeated references to one key. |
| `duplicate_rows_reused` | Additional uncached rows sharing an input encoded within this call. |
| `encoded_inputs` | Input rows encoded on the final execution path; discarded retry work is not accumulated. Cache-key reuse deduplicates repeated inputs when caching is active; without it, repeated inputs are encoded separately. |
| `model_loaded` | Whether the corpus call needed the model, including a previously loaded process-local instance. |
| `cache_enabled` | Whether persistent reuse was enabled for the call; writes can still fail. |
| `cache_warnings` | Non-fatal cache read/write, manifest, and query-cache failures observed during the run. |
| `cache_revision` | Revision label, commit, or local fingerprint used for cache identity, otherwise `null`. |
| `execution_device` | Effective inference device, or `null` when no model execution was needed. |
| `moved_units_reused` | New UIDs matched to departed UIDs with the same content after retaining same-file symbols whose byte offsets changed. |
| `deleted_units` | Departed UIDs left after matching retained symbols and moves, independent of vector sharing. |
| `orphan_rows_retained` | Tracked orphan rows still stored, including rows protected by another active selection. |
| `orphan_rows_collected` | Orphan rows removed during this run. |
| `manifest_generation` | Shard-wide complete-scan counter, or `null` without successful manifest publication. |

Move and deletion counts need a comparable [corpus baseline](caching.md#corpus-lifecycle). Terminal `check` and `search` output include an `Embeddings` summary showing cache hits, encoded inputs, duplicate-row reuse, model execution, and manifest generation. Nonzero move, deletion, and orphan counts are included too.

## Diagnostics

`check` emits `extraction_diagnostics` and `semantic_diagnostics` arrays with matching counts in `summary`. `search` emits both diagnostic arrays, without summary counts, so recoverable extraction failures remain visible even when the search index is empty. Entries use `file`, `language`, `severity`, `code`, `message`, `line`, and `end_line`. Terminal checks print counts and up to ten entries per diagnostic category; terminal searches print semantic diagnostics.

For `semantic-context-overflow` warnings and their cache behavior, see [long-input handling](analysis-defaults.md#semantic-candidate-defaults).

## Exit codes

`check --fail-on` controls findings only; runtime and usage failures retain their normal status:

- `--fail-on actionable` (default): combined mode exits `1` for `exact`, `traditional_near`, or `hybrid_confirmed`. Pure-semantic `semantic_high_confidence` pairs are reported but advisory, and `semantic_review` pairs are withheld and advisory, because neither has deterministic structural/token corroboration. Non-strict unused guesses are also advisory, while `--strict-unused` makes them actionable. Raw single-method duplicates already passed the explicitly selected method thresholds and remain actionable.
- `--fail-on all`: any duplicate or unused finding in the complete result exits `1`, including `semantic_review` pairs the report withheld.
- `--fail-on none`: findings never change the successful exit code.

The exit code is computed on the complete analysis result before report selection, so `--include-review`, `--show-all`, and `--max-duplicates` never change it. When every failing finding is hidden from the report, the terminal `Finding status` row says which hidden group fails and how to list it: withheld review pairs under `--fail-on all` point at `--include-review`, and pairs cut by `--max-duplicates` point at a higher cap (a strong pure-semantic pair can outrank a corroborated one, so a small cap can hide the only `hybrid_confirmed` pair that fails `actionable`). In JSON the same situations read as `exit_code: 1` with no failing edge in `duplicates` and a non-zero `omitted_review_duplicates` or `truncated_duplicates`. The selected policy and computed result are always present as `summary.fail_on` and `summary.exit_code`. Terminal summaries show the same values as `Failure policy` and `Finding status` rows.

Command status conventions:

- `0`: command completed and the selected finding policy did not fail the run.
- `1`: selected findings failed `check`, or a command encountered a runtime failure.
- `2`: Click usage or validation error.

Default combined semantic failures are fatal. `--allow-semantic-fallback` continues with full-scope traditional results and records `summary.semantic_fallback` plus `summary.semantic_fallback_reason`; under the default actionable policy, heuristic unused findings alone do not turn that successful degraded run into exit `1`.

## Terminal duplicate panels

Tables show up to 20 rows by default; `--full-table` removes that presentation limit. The footer counts additional selected pairs. The [report-level cap](#report-selection) applies before table rendering.

Locations use the shorter of working-directory-relative and absolute `<path>:<line>` spellings.

- Combined: `Hybrid Duplicates (N pairs, M review withheld, K truncated)`, followed by any raw panels requested through [report selection](#report-selection). When every hybrid pair is withheld, one dim line reports the withheld count instead of an empty table. The summary lists every tier's count; withheld and truncated totals appear when non-zero.
- `--traditional-only`: `Traditional Duplicates (Structural/Token/Jaccard)`.
- `--semantic-only`: `Semantic Duplicates (Embedding)`.
