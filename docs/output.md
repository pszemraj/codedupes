# Output and exit codes

stdout carries report output only: JSON under `--json`, Rich tables otherwise. Errors and parser-unavailable remediation use stderr; Rich mode also sends logs, cache warnings, sentence-transformers progress, and Hugging Face download progress there. JSON mode disables progress and records non-fatal cache failures in `summary.embeddings.cache_warnings` instead of emitting them, so merged streams remain parseable:

```text
codedupes check ./src --json --no-cache 2>&1 | python -c "import json,sys; json.load(sys.stdin)"
```

## JSON schema v2

`check --json` and `search --json` emit schema version `2`. Units are nodes in a top-level `units` object keyed by `CodeUnit.uid`; findings refer to those keys instead of repeating a complete unit object for every pair endpoint.

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

The shortened example omits the other two referenced entries from `units`; real output includes every UID referenced by `duplicates` or `potentially_unused` exactly once. Units with no finding are not emitted; `summary.total_units` is the full extracted corpus count.

In default combined mode, `duplicates` contains hybrid edges. With `--show-all`, `traditional_duplicates` and `semantic_duplicates` are added as raw edge lists with `unit_a`, `unit_b`, `similarity`, and `method`.

In `--semantic-only` or `--traditional-only` mode, `duplicates` directly contains the active raw edge list and the `--show-all` arrays are omitted. `analysis_mode` is always one of `combined`, `traditional`, `semantic`, or `none`.

See [hybrid confidence tiers](analysis-defaults.md#hybrid-synthesis-confidence-defaults) to interpret `tier` and `confidence`.

### Search

```json
{
  "schema_version": 2,
  "query": "refund validation",
  "summary": {
    "indexed_units": 42,
    "results": 1,
    "embeddings": null
  },
  "results": [
    {
      "unit": "/repo/billing/refunds.py::python::billing.refunds.validate::0",
      "score": 0.95
    }
  ],
  "units": {
    "/repo/billing/refunds.py::python::billing.refunds.validate::0": {
      "name": "validate",
      "qualified_name": "billing.refunds.validate",
      "type": "function",
      "language": "python",
      "dialect": "python",
      "native_kind": "FunctionDef",
      "file": "/repo/billing/refunds.py",
      "line": 1,
      "end_line": 2,
      "start_byte": 0,
      "end_byte": 42,
      "start_column": 0,
      "end_column": 20,
      "statement_count": 1,
      "is_public": true,
      "is_exported": false
    }
  },
  "semantic_diagnostics": []
}
```

`summary.indexed_units` is the semantic corpus size after eligibility and context-window filtering. An empty terminal index warns on stderr and distinguishes empty extraction, eligibility filtering, and context-window exclusions.

## Embedding telemetry

`summary.embeddings` describes the final retained corpus from the most recent embedding call; it is `null` when semantic work did not run or fell back. Query encoding does not increment these counters, though query-cache warnings are appended. A warm corpus can therefore report `model_loaded: false` while a new query still loads the model.

| Field | Meaning |
| --- | --- |
| `requested_rows` | Corpus rows retained after context-window filtering. |
| `unique_inputs` | Distinct prepared texts among those rows. |
| `cache_hit_rows` | Retained rows supplied by persistent cache, including repeated references to one key. |
| `duplicate_rows_reused` | Additional uncached rows sharing an input encoded within this call. |
| `encoded_inputs` | Distinct retained inputs encoded on the final execution path; discarded retry work is not accumulated. |
| `model_loaded` | Whether the corpus call needed the model, including a previously loaded process-local instance. |
| `cache_enabled` | Whether persistent reuse was enabled for the call; writes can still fail. |
| `cache_warnings` | Non-fatal cache read/write, manifest, and query-cache failures observed during the run. |
| `cache_revision` | Revision label, commit, or local fingerprint used for cache identity, otherwise `null`. |
| `execution_device` | Effective inference device, or `null` when no model execution was needed. |
| `moved_units_reused` | New UIDs matched to departed UIDs with the same content. |
| `deleted_units` | Departed UIDs left after matching moves, independent of vector sharing. |
| `orphan_rows_retained` | Tracked orphan rows still stored, including rows protected by another active selection. |
| `orphan_rows_collected` | Orphan rows removed during this run. |
| `manifest_generation` | Shard-wide complete-scan counter, or `null` without successful manifest publication. |

Move and deletion counts need a comparable [corpus baseline](caching.md#corpus-lifecycle). Terminal `check` output shows the same statistics on an `Embeddings` summary row.

## Diagnostics

`check` emits `extraction_diagnostics` and `semantic_diagnostics` arrays with matching counts in `summary`. `search` emits only the semantic diagnostic array, without a summary count; its JSON does not include extraction diagnostics. Entries use `file`, `language`, `severity`, `code`, `message`, `line`, and `end_line`. Terminal checks print counts and up to ten entries per diagnostic category; terminal searches print semantic diagnostics.

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
