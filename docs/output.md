# Output and exit codes

## Report streams

For `check` and `search`, stdout contains the report: JSON under `--json` and Rich
tables otherwise. Errors and parser-unavailable remediation use stderr; Rich mode also
sends logs, cache warnings, sentence-transformers progress, and Hugging Face download
progress there. A completed JSON report is a single parseable JSON document even when
`check` exits `1` for findings. JSON mode disables progress and records non-fatal cache
failures in `summary.embeddings.cache_warnings` instead of emitting them. Runtime
failures restore stderr and do not produce a completed JSON report.

Write a JSON report directly in automation:

```text
codedupes check ./src --json > codedupes-report.json
```

On a completed scan, the report is written before the command returns its finding
status. With the default `--fail-on actionable`, a completed report with exit `1`
contains an actionable finding. Runtime failures also exit `1`, but leave no completed
report and explain the error on stderr. Use `--fail-on none` to collect a report
without making findings fail an incremental rollout:

```text
codedupes check ./src --json --fail-on none > codedupes-report.json
```

If a Bash or Zsh pipeline validates JSON with `jq`, enable `pipefail` so the parser's successful
exit does not hide `codedupes`' status:

```text
set -o pipefail
codedupes check ./src --json | jq empty
```

## JSON schema v2

`check --json` and `search --json` emit schema version `2`. Units are nodes in a
top-level `units` object keyed by `CodeUnit.uid`; findings refer to those keys instead
of repeating a complete unit object for every pair endpoint. A UID is unique within one
report and includes the source path and byte position, so use it to join data within
that report rather than as a cross-machine finding identifier.

### Check

```json
{
  "schema_version": 2,
  "analysis_mode": "combined",
  "summary": {
    "total_units": 42,
    "units_by_language": {"python": 42},
    "hybrid_duplicates": 1,
    "potentially_unused": 1,
    "raw_traditional_duplicates": 1,
    "raw_semantic_duplicates": 1,
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
      "unit_a": "/repo/src/a.py::python::a.normalize::0",
      "unit_b": "/repo/src/b.py::python::b.normalize::0",
      "tier": "hybrid_confirmed",
      "confidence": 0.94,
      "has_exact": false,
      "semantic_similarity": 0.96,
      "jaccard_similarity": 0.92,
      "weak_identifier_jaccard": 0.7,
      "statement_count_ratio": 1.0
    }
  ],
  "potentially_unused": [
    "/repo/src/unused.py::python::unused.helper::0"
  ],
  "extraction_diagnostics": [],
  "semantic_diagnostics": [],
  "units": {
    "/repo/src/a.py::python::a.normalize::0": {
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

The shortened example omits the other two referenced entries from `units`; real output
includes every UID referenced by any finding list exactly once. Units with no finding
are not emitted, so `summary.total_units` is the full extracted corpus count while
`units` contains only units needed to resolve reported findings.

In default combined mode, `duplicates` contains hybrid edges. With `--show-all`, `traditional_duplicates` and `semantic_duplicates` are added as raw edge lists with `unit_a`, `unit_b`, `similarity`, and `method`.

In `--semantic-only` or `--traditional-only` mode, `duplicates` directly contains the active raw edge list and the `--show-all` arrays are omitted. `analysis_mode` is always one of `combined`, `traditional`, `semantic`, or `none`.

See [hybrid confidence tiers](analysis-defaults.md#hybrid-synthesis-confidence-defaults) to interpret `tier` and `confidence`.

### Search

Default search hits (`--result-level unit`) use `{"unit": "<uid>", "score": 0.95}`; their unit records have the same fields as check results. An empty index with `--no-cache` produces:

```json
{
  "schema_version": 2,
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
    {"unit": "/repo/src/parser.py::python::parser.parse::0", "score": 0.91},
    {"unit": "/repo/src/parser.py::python::parser.decode::240", "score": 0.86}
  ]
}
```

The file's `score` is its highest unit score. `matching_units` counts all of that file's units above the search threshold; `matches` contains up to three strongest contributors. Their UIDs reference the top-level `units` map, which supplies names, types, and line ranges. Only these displayed contributors appear in `units`. Files are ranked before applying `--top-k`; `summary.results` counts returned files, while `summary.indexed_units` still counts indexed code units. Diagnostics and embedding telemetry keep the same shape. Unit-level output remains the default and does not add `result_level`.

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

`semantic-context-overflow` warns that a newly encoded unit exceeds the model's context window and will be truncated by the backend. It remains in results. These warnings also cover units re-encoded after incompatible cached vectors are discarded. They are produced during corpus inference, not replayed on reused cache hits; see [long-input behavior](analysis-defaults.md#semantic-candidate-defaults).

## Exit codes

`check --fail-on` controls findings only; runtime and usage failures retain their normal status:

- `--fail-on actionable` (default): combined mode exits `1` for `exact`, `traditional_near`, or `hybrid_confirmed`. Pure-semantic `semantic_high_confidence` and `semantic_review` pairs remain visible but advisory because neither has deterministic structural/token corroboration. Non-strict unused guesses are also advisory, while `--strict-unused` makes them actionable. Raw single-method duplicates already passed the explicitly selected method thresholds and remain actionable.
- `--fail-on all`: any reported duplicate or unused finding exits `1`.
- `--fail-on none`: findings never change the successful exit code.

The selected policy and computed result are always present as `summary.fail_on` and `summary.exit_code`. Terminal summaries show the same values as `Failure policy` and `Finding status` rows.

Command status conventions:

- `0`: command completed and the selected finding policy did not fail the run.
- `1`: selected findings failed `check`, or a command encountered a runtime failure.
- `2`: Click usage or validation error.

Default combined semantic failures are fatal. `--allow-semantic-fallback` continues with full-scope traditional results and records `summary.semantic_fallback` plus `summary.semantic_fallback_reason`; under the default actionable policy, heuristic unused findings alone do not turn that successful degraded run into exit `1`.

## Terminal duplicate panels

Tables show up to 20 rows by default; `--full-table` removes that limit.

Locations use the shorter of working-directory-relative and absolute `<path>:<line>` spellings.

- Combined: `Hybrid Duplicates`, plus raw traditional and semantic panels under `--show-all`.
- `--traditional-only`: `Traditional Duplicates (Structural/Token/Jaccard)`.
- `--semantic-only`: `Semantic Duplicates (Embedding)`.
