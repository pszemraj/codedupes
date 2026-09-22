# Output and exit codes

## Report streams

For `check` and `search`, stdout contains the report: JSON under `--json` and Rich
tables otherwise. Errors and parser-unavailable remediation use stderr; Rich mode also
sends logs, cache warnings, sentence-transformers progress, and Hugging Face download
progress there. A completed JSON report is a single parseable JSON document even when
`check` exits `1` for findings. JSON mode disables progress and records non-fatal cache
failures in `summary.embeddings.cache_warnings` instead of emitting them. Direct
backend output on stdout or stderr, including buffered C stdio, is captured during
JSON runs and replayed to stderr only on runtime failure, without a completed JSON report.

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

## JSON schema v4

`check --json` and `search --json` emit schema version `4`. Units are nodes in a top-level `units` object keyed by report-local ids (`u0`, `u1`, ...); findings refer to those ids instead of repeating a complete unit object for every endpoint. Ids are assigned in file-path then source-offset order over the referenced units only, so they renumber whenever the referenced set changes (for example with `--include-review`). Treat them as opaque within one report. Each unit record also carries the [in-run `CodeUnit.uid`](python-api.md#key-result-types).

### Check

```json
{
  "schema_version": 4,
  "analysis_mode": "combined",
  "summary": {
    "total_units": 42,
    "units_by_language": {"python": 42},
    "hybrid_duplicates": 4,
    "reported_duplicates": 2,
    "omitted_review_duplicates": 2,
    "truncated_duplicates": 0,
    "truncated_by_tier": {
      "exact": 0,
      "traditional_near": 0,
      "hybrid_confirmed": 0,
      "semantic_high_confidence": 0,
      "semantic_review": 0
    },
    "max_duplicates": 20,
    "actionable_duplicates": 2,
    "reported_actionable_duplicates": 2,
    "duplicates_by_tier": {
      "exact": 1,
      "traditional_near": 0,
      "hybrid_confirmed": 1,
      "semantic_high_confidence": 0,
      "semantic_review": 2
    },
    "exact_family_members": 3,
    "potentially_unused": 1,
    "reported_unused": 1,
    "truncated_unused": 0,
    "max_unused": 20,
    "raw_traditional_duplicates": 4,
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
      "execution_device": "cuda",
      "moved_units_reused": 0,
      "deleted_units": 0,
      "orphan_rows_retained": 2,
      "orphan_rows_collected": 0,
      "manifest_generation": 17
    },
    "fail_on": "actionable",
    "strict_unused": false,
    "exit_code": 1,
    "hidden_only_failure": []
  },
  "exact_families": [
    {
      "method": "token_hash",
      "members": ["u3", "u4", "u5"],
      "lines": 18,
      "redundant_lines": 36
    }
  ],
  "duplicates": [
    {
      "unit_a": "u0",
      "unit_b": "u1",
      "tier": "hybrid_confirmed",
      "score": 0.94,
      "semantic_similarity": 0.96,
      "jaccard_similarity": 0.92,
      "weak_identifier_jaccard": null,
      "statement_count_ratio": null
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
      "native_kind": "function_definition",
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

The shortened example omits `u1` through `u5` from `units`; real output includes every id referenced by any emitted finding list exactly once. Units with no emitted finding are not present, so `summary.total_units` is the full extracted corpus count while `units` contains only units needed to resolve the report.

`native_kind` is the grammar node kind (`function_definition` or `class_definition` for Python, whether or not the definition is decorated). `line`, `start_byte`, and `start_column` locate the unit's first byte - the first decorator of a decorated Python definition - so an indented method reports a non-zero `start_column`; see [source ranges](polyglot-languages.md#source-ranges-and-parse-recovery).

`weak_identifier_jaccard` and `statement_count_ratio` are computed only for the two semantic-only tiers; they are `null` for traditional-near and hybrid-confirmed pairs.

#### Exact families

Exact duplicates are an equivalence, not a scored pair, so the report groups them: a set of `n` mutually identical units is one `exact_families` record instead of `n(n-1)/2` `exact` edges, and `duplicates` never contains an exact edge. `members` lists the report ids in file order, `lines` is the line span of the largest member, and `redundant_lines` is `(members - 1) * lines`, the source you would delete by keeping one copy. `method` is the strongest fingerprint every member shares: `token_hash` members are token-for-token copies (comments and whitespace aside), while `structural_hash` members match only after identifier and literal normalization, so they may differ in names and string literals. Families are built per fingerprint from the exact edges of the complete result, so they are the same in every mode and under every cap.

#### Report selection

In default combined mode the primary list is `exact_families` followed by `duplicates`, which holds hybrid edges of every tier except `semantic_review`; see [tier evidence](analysis-defaults.md#hybrid-synthesis-confidence-defaults). The list is ranked for review, not by raw score: families come first in descending `redundant_lines` (ties by first-member position), then the actionable pair tiers (`traditional_near`, `hybrid_confirmed`), then `semantic_high_confidence`, then any `semantic_review` pairs admitted by `--include-review`; inside each pair group pairs keep the analyzer's [score order](analysis-defaults.md#score-scale). `--show-all` implies `--include-review` and also adds `traditional_duplicates` and `semantic_duplicates` as raw edge lists with `unit_a`, `unit_b`, `similarity`, and `method` (`structural_hash`, `token_hash`, or `jaccard` for traditional edges; `semantic` for semantic edges); those raw lists still spell out every pairwise exact edge.

Summary counts are findings, where a family counts once and every other tier counts per pair. `summary.hybrid_duplicates` counts the complete synthesis that way, `summary.duplicates_by_tier` breaks it down over all five tiers (always present, zero-filled; `exact` is the family count), `summary.exact_family_members` counts the distinct units inside families, `summary.reported_duplicates` counts the families and pairs actually emitted, `summary.omitted_review_duplicates` counts pairs withheld by the report policy, and `summary.truncated_duplicates` counts findings cut by the report cap, broken down over the same five tiers in `summary.truncated_by_tier` (`exact` is the number of cut families). In combined mode, `reported_duplicates + omitted_review_duplicates + truncated_duplicates == hybrid_duplicates` regardless of report-selection flags. `summary.actionable_duplicates` counts the findings in the complete result that fail `--fail-on actionable` (families plus actionable tiers in combined mode; families plus every raw pair in a single-method mode) and `summary.reported_actionable_duplicates` counts how many of those are emitted, so `reported_duplicates - reported_actionable_duplicates` is the number of advisory pairs on the report. `summary.raw_traditional_duplicates` and `summary.raw_semantic_duplicates` still count raw edges.

The primary list is capped by default: families plus pairs hold at most 20 findings (`--max-duplicates`, recorded as `summary.max_duplicates`), the same findings in the same order as the terminal panels, and `units` drops units referenced only by cut findings. Because families and actionable tiers rank first, the cap trims advisory candidates before corroborated ones, and a family is one item however many copies it holds. `--max-duplicates N` changes the budget and `--max-duplicates all` removes it (`max_duplicates: null`); `--include-review`, `--show-all`, and `--full-table` also remove it unless an explicit `--max-duplicates` accompanies them. With `--include-review --max-duplicates N`, review pairs sit at the end of the ranking, so they appear only once every actionable and advisory finding fits under `N`; findings the cap cuts count as `truncated`, not `omitted_review`, and `truncated_by_tier.semantic_review` says how many review pairs were cut. The raw `--show-all` lists are never capped. The exit code ignores the cap, see [exit codes](#exit-codes).

In `--semantic-only` or `--traditional-only` mode, exact edges are grouped into `exact_families` the same way and `duplicates` contains the remaining raw edges ordered by descending similarity (ties in analyzer order; the cap keeps families then that prefix). `duplicates_by_tier` and `truncated_by_tier` are zero except for `exact`, which counts families, `hybrid_duplicates` is `0`, and the `--show-all` arrays are omitted. `analysis_mode` is always one of `combined`, `traditional`, `semantic`, or `none`.

`potentially_unused` is ranked and bounded too: ids are ordered by line span (`end_line - line + 1`) descending, then statement count, then file position, so the largest dead definitions lead, and the list holds at most 20 (`--max-unused`, recorded as `summary.max_unused`; `--max-unused all` removes the cap and the expansion flags above lift it unless an explicit value is given). `summary.potentially_unused` stays the complete count while `summary.reported_unused` and `summary.truncated_unused` split it into emitted and cut; units referenced only by cut unused findings leave `units`. `extraction_diagnostics`, `semantic_diagnostics`, and the raw `--show-all` edge lists are deliberately complete: they are per-file records a consumer needs in full, so a scan with many diagnostics still produces a large document.

See [hybrid confidence tiers](analysis-defaults.md#hybrid-synthesis-confidence-defaults) to interpret `tier` and `score`.

### Search

Default search hits (`--result-level unit`) use `{"unit": "u0", "score": 0.95}`; their unit records have the same fields as check results. An empty index with `--no-cache` produces:

```json
{
  "schema_version": 4,
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
| `execution_device` | Effective `cpu`, `cuda`, or `mps` inference device, or `null` when no model execution was needed. |
| `moved_units_reused` | New UIDs matched to departed UIDs with the same content after retaining same-file symbols whose byte offsets changed. |
| `deleted_units` | Departed UIDs left after matching retained symbols and moves, independent of vector sharing. |
| `orphan_rows_retained` | Tracked orphan rows still stored, including rows protected by another active selection. |
| `orphan_rows_collected` | Orphan rows removed during this run. |
| `manifest_generation` | Shard-wide complete-scan counter, or `null` without successful manifest publication. |

Move and deletion counts need a comparable [corpus baseline](caching.md#corpus-lifecycle). Terminal `check` and `search` output include an `Embeddings` summary showing cache hits, encoded inputs, duplicate-row reuse, model execution, and manifest generation. Nonzero move, deletion, and orphan counts are included too.

## Diagnostics

`check` emits `extraction_diagnostics` and `semantic_diagnostics` arrays with matching counts in `summary`. `search` emits both diagnostic arrays, without summary counts, so recoverable extraction failures remain visible even when the search index is empty. Entries use `file`, `language`, `severity`, `code`, `message`, `line`, and `end_line`. Both terminal commands print up to ten entries per diagnostic category; checks also print summary counts.

For `semantic-context-overflow` warnings and their cache behavior, see [long-input handling](analysis-defaults.md#semantic-candidate-defaults).

## Exit codes

`check --fail-on` controls findings only; runtime and usage failures retain their normal status:

- `--fail-on actionable` (default): combined mode exits `1` for `exact`, `traditional_near`, or `hybrid_confirmed`. Pure-semantic `semantic_high_confidence` pairs are reported but advisory, and `semantic_review` pairs are withheld and advisory, because neither has deterministic structural/token corroboration. Non-strict unused guesses are also advisory, while `--strict-unused` makes them actionable. Raw single-method duplicates already passed the explicitly selected method thresholds and remain actionable.
- `--fail-on all`: any duplicate or unused finding in the complete result exits `1`, including `semantic_review` pairs the report withheld.
- `--fail-on none`: findings never change the successful exit code.

The exit code is computed on the complete analysis result before report selection, so `--include-review`, `--show-all`, `--max-duplicates`, and `--max-unused` never change it. The only way every failing finding can be hidden from the report is `--fail-on all` with withheld `semantic_review` pairs; the terminal `Finding status` row then says so and points at `--include-review`, and JSON records `"hidden_only_failure": ["review"]` (otherwise `[]`). The report caps cannot cause this: families and actionable tiers rank first and unused failure is all-or-nothing, so whenever a cut finding fails, an emitted finding fails too. The selected policy, the unused strictness it was evaluated with, and the computed result are always present as `summary.fail_on`, `summary.strict_unused`, and `summary.exit_code`. Terminal summaries show the same values as `Failure policy` and `Finding status` rows.

Command status conventions:

- `0`: command completed and the selected finding policy did not fail the run.
- `1`: selected findings failed `check`, or a command encountered a runtime failure.
- `2`: CLI usage or validation error.

Default combined semantic backend or runtime failures are fatal. `--allow-semantic-fallback` continues with full-scope traditional results and records `summary.semantic_fallback` plus `summary.semantic_fallback_reason`; under the default actionable policy, heuristic unused findings alone do not turn that successful degraded run into exit `1`.

## Terminal duplicate panels

The primary panels list every finding the [report caps](#report-selection) selected, so they show exactly what `--json` would emit. `Exact Duplicate Families (N families, K truncated)` comes first when any family is kept, one row per family with member count, `Lines`, `Method`, the first member, and up to three more locations before a `+N more` note; `--show-source` prints one snippet per member. The pair table follows, and the unused table (`Likely Dead Code (N units, K truncated)` in combined mode, `Potentially Unused` otherwise) lists the largest units first with a `Lines` column. Only the raw `--show-all` tables keep a 20-row display limit; their footers point to `--full-table`, which lifts that limit and, unless given explicitly, the `--max-duplicates` and `--max-unused` caps as well.

Locations use the shorter of working-directory-relative and absolute `<path>:<line>` spellings.

- Combined: `Hybrid Duplicates (N pairs, M review withheld, K truncated)`, followed by any raw panels requested through [report selection](#report-selection). When every hybrid pair is withheld, one dim line reports the withheld count instead of an empty table. The summary lists every tier's count, the `exact` row reading `N families (M units)`, plus `Actionable duplicates` as `total (reported)`; withheld and truncated totals appear when non-zero, the latter naming the cut tiers (families as `N exact families`) and pointing at `--max-duplicates all`, and `Truncated dead code` points at `--max-unused all`.
- `--traditional-only`: `Traditional Duplicates (Structural/Token/Jaccard)` for the non-exact pairs, after the family panel; the summary adds an `Exact duplicate families` row.
- `--semantic-only`: `Semantic Duplicates (Embedding)`.
