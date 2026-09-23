# Output and exit codes

## Report streams

For `check` and `search`, stdout contains the report: JSON under `--json` and Rich tables otherwise. Errors and parser-unavailable remediation use stderr; Rich mode also sends logs, cache warnings, sentence-transformers progress, and Hugging Face download progress there. A completed JSON report is a single parseable JSON document even when `check` exits `1` for findings (or an incomplete analysis under `--fail-on-incomplete`). JSON mode disables progress and records non-fatal cache failures in `summary.embeddings.cache_warnings` instead of emitting them. Direct backend output on stdout or stderr, including buffered C stdio, is captured during JSON runs and replayed to stderr only on a runtime failure (exit `3`), without a completed JSON report.

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
  "analysis_status": "complete",
  "run": {
    "tool_version": "0.9.0",
    "root": "/repo",
    "target": "/repo",
    "languages": null,
    "exclude_patterns": ["**/.git/**", "**/node_modules/**"],
    "respect_gitignore": true,
    "include_private": true,
    "include_stubs": false,
    "extracted_files": 40,
    "units": {
      "extracted": 42,
      "semantic_eligible": 40,
      "by_language": {"python": 42},
      "by_type": {"class": 6, "function": 24, "method": 12}
    },
    "traditional": {
      "jaccard_threshold": 0.85,
      "tiny_filter": true,
      "tiny_cutoff": 3
    },
    "semantic": {
      "requested_model": "Alibaba-NLP/gte-modernbert-base",
      "model": "Alibaba-NLP/gte-modernbert-base",
      "revision": "abc1234",
      "profile": "gte-modernbert",
      "threshold_profile": "auto",
      "task": "code-duplicate",
      "device": "auto",
      "execution_device": "cuda",
      "thresholds": {"python": 0.86},
      "threshold_floor": 0.6,
      "min_statements": 3,
      "unit_types": ["function", "method"],
      "cross_language": false,
      "hybrid_split": {
        "weak_identifier_jaccard_min": 0.4,
        "statement_ratio_min": 0.6,
        "promotion_gates": {"python": 0.9}
      }
    },
    "unused": {"strict": false, "files": 40},
    "checks": {
      "extraction": {"status": "completed", "files": 40, "files_failed": 0, "diagnostics": 0},
      "traditional": {"status": "completed", "files": null, "files_failed": 0, "diagnostics": 0},
      "semantic": {"status": "completed", "files": null, "files_failed": 0, "diagnostics": 0},
      "unused": {"status": "completed", "files": 40, "files_failed": 0, "diagnostics": 0}
    }
  },
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
    "unused_supported_languages": ["python"],
    "unused_excluded_units": 0,
    "suppressed_duplicates": 0,
    "suppressed_unused": 0,
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
    "fail_on_incomplete": false,
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
  "unused_diagnostics": [],
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

`native_kind` is the grammar node kind (`function_definition` or `class_definition` for Python, whether or not the definition is decorated). `line`, `start_byte`, and `start_column` locate the unit's first byte - the first decorator of a decorated Python definition - so an indented method reports a non-zero `start_column`; see [source ranges](polyglot-languages.md#source-ranges-and-parse-recovery). `suppressions` is opt-in: present only on a unit carrying a [`codedupes: ignore` directive](analysis-defaults.md#suppression-directives), sorted, and never an empty list.

`--show-source` (or an explicit `--source-lines`) adds `source` and `source_lines_omitted` to every unit record: `source` is the unit's source joined back with `\n`, bounded to `--source-lines` lines (default `40`, `all` removes the bound), and `source_lines_omitted` is the count of trailing lines cut to fit that bound (`0` when nothing was cut). Neither field is present without `--show-source`/`--source-lines`, and every other default payload stays byte-identical.

`weak_identifier_jaccard` and `statement_count_ratio` are computed only for the two semantic-only tiers; they are `null` for traditional-near and hybrid-confirmed pairs.

#### Run record and check status

`run` is what this analysis actually configured and did, independent of what it found: `root` is the resolved analysis root, `target` preserves an explicit file or symlink target's own path, `exclude_patterns` is the effective exclude list after default resolution, and `include_stubs` is true for an explicitly selected `.pyi` file even when directory discovery excludes stubs. `units.extracted`/`units.semantic_eligible` count the corpus before and after semantic candidate filtering. `units.by_language` and `units.by_type` break the extracted corpus down and are what the terminal summary prints: `by_type` always carries `class`, `function`, and `method` (zero when absent), while `by_language` lists only the languages actually extracted. `traditional`, `semantic`, and `unused` are `null` when that detector did not run and otherwise carry its resolved settings — `semantic.model`/`semantic.revision` reflect what actually loaded (falling back to the requested name/revision when no model load was needed), `semantic.execution_device` is the device the model last ran on for this analysis and is `null` when every embedding came from cache, and `semantic.hybrid_split` is `null` outside combined mode.

`run.checks` derives one status per detector from `run` and this result's diagnostics, so it never needs its own storage: `extraction` is `empty` when the corpus has no units, `partial` when any file raised a scope-losing diagnostic (`read-error`, `invalid-utf8`, `partial-parse`, `unit-parse-error`, `walk-error`) and `completed` otherwise — an advisory-only diagnostic (`c-header-policy`, `semantic-context-overflow`, `suppression-syntax`) does not mark extraction partial. `traditional` and `unused` are `disabled` when that detector did not run, else `completed` (`unused` is `partial` when any file raised an unused-analysis diagnostic). `semantic` is `disabled` when it did not run, `fallback` when combined mode degraded to traditional-only results (`summary.semantic_fallback`), else `completed`. Each check record's `files`/`files_failed`/`diagnostics` count that detector's own scope; `files` is `null` for `traditional` and `semantic`, which do not have a per-file failure count.

`analysis_status` (also `AnalysisChecks.analysis_status` in the Python API) summarizes `run.checks` in one value: `"empty"` when extraction produced no units, `"partial"` when extraction is `partial`, semantic fell back, or unused is `partial`, otherwise `"complete"`. A `"complete"` scan can still report zero duplicates and zero unused findings — this field is about whether every configured check ran to completion, not about what it found.

#### Exact families

Exact duplicates are an equivalence, not a scored pair, so the report groups them: a set of `n` mutually identical units is one `exact_families` record instead of `n(n-1)/2` `exact` edges, and `duplicates` never contains an exact edge. `members` lists the report ids in file order, `lines` is the line span of the largest member, and `redundant_lines` is the sum of all member spans minus that largest span: the source lines removable if the largest copy is kept. `method` is the strongest fingerprint every member shares: `token_hash` members are token-for-token copies (comments and whitespace aside), while `structural_hash` members match only after identifier and string-literal normalization, so they may differ in names and string literals — numeric literals are not normalized, so a renamed pair still keeps the same numbers in both bodies. A token clique whose members are all also structurally equal to a renamed relative folds into that larger `structural_hash` family instead of keeping its own record; a token component that is not fully covered by a structural family (Python indentation can make units token-equal without making them structurally equal) forms its own `token_hash` family. Families are built per fingerprint from the exact edges of the complete result, so they are the same in every mode and under every cap.

#### Report selection

In default combined mode the primary list is `exact_families` followed by `duplicates`, which holds hybrid edges of every tier except `semantic_review`; see [tier evidence](analysis-defaults.md#hybrid-synthesis-confidence-defaults). The list is ranked for review, not by raw score: families come first in descending `redundant_lines` (ties by first-member position), then the actionable pair tiers (`traditional_near`, `hybrid_confirmed`), then `semantic_high_confidence`, then any `semantic_review` pairs admitted by `--include-review`; inside each pair group pairs keep the analyzer's [score order](analysis-defaults.md#score-scale). `--show-all` implies `--include-review` and also adds `traditional_duplicates` and `semantic_duplicates` as raw edge lists with `unit_a`, `unit_b`, `similarity`, and `method` (`structural_hash`, `token_hash`, or `jaccard` for traditional edges; `semantic` for semantic edges); those raw lists still spell out every pairwise exact edge.

Summary counts are findings, where a family counts once and every other tier counts per pair. `summary.hybrid_duplicates` counts the complete synthesis that way, `summary.duplicates_by_tier` breaks it down over all five tiers (always present, zero-filled; `exact` is the family count), `summary.exact_family_members` counts the distinct units inside families, `summary.reported_duplicates` counts the families and pairs actually emitted, `summary.omitted_review_duplicates` counts pairs withheld by the report policy, and `summary.truncated_duplicates` counts findings cut by the report cap, broken down over the same five tiers in `summary.truncated_by_tier` (`exact` is the number of cut families). In combined mode, `reported_duplicates + omitted_review_duplicates + truncated_duplicates == hybrid_duplicates` regardless of report-selection flags. `summary.actionable_duplicates` counts the findings in the complete result that fail `--fail-on actionable` (families plus actionable tiers in combined mode; families plus every raw pair in a single-method mode) and `summary.reported_actionable_duplicates` counts how many of those are emitted, so `reported_duplicates - reported_actionable_duplicates` is the number of advisory pairs on the report. `summary.raw_traditional_duplicates` and `summary.raw_semantic_duplicates` still count raw edges, after `summary.suppressed_duplicates` pairs were dropped for carrying a `codedupes: ignore[duplicates]` directive on either endpoint (see [suppression directives](analysis-defaults.md#suppression-directives)); those pairs never reach families, raw edges, or hybrid synthesis.

The primary list is capped by default: families plus pairs hold at most 20 findings (`--max-duplicates`, recorded as `summary.max_duplicates`), the same findings in the same order as the terminal panels, and `units` drops units referenced only by cut findings. Because families and actionable tiers rank first, the cap trims advisory candidates before corroborated ones, and a family is one item however many copies it holds. `--max-duplicates N` changes the budget and `--max-duplicates all` removes it (`max_duplicates: null`); `--include-review`, `--show-all`, and `--full-table` also remove it unless an explicit `--max-duplicates` accompanies them. With `--include-review --max-duplicates N`, review pairs sit at the end of the ranking, so they appear only once every actionable and advisory finding fits under `N`; findings the cap cuts count as `truncated`, not `omitted_review`, and `truncated_by_tier.semantic_review` says how many review pairs were cut. The raw `--show-all` lists are never capped. The exit code ignores the cap, see [exit codes](#exit-codes).

In `--semantic-only` or `--traditional-only` mode, exact edges are grouped into `exact_families` the same way and `duplicates` contains the remaining raw edges ordered by descending similarity (ties in analyzer order; the cap keeps families then that prefix). `duplicates_by_tier` and `truncated_by_tier` are zero except for `exact`, which counts families, `hybrid_duplicates` is `0`, and the `--show-all` arrays are omitted. `analysis_mode` is always one of `combined`, `traditional`, `semantic`, or `unused` (neither traditional nor semantic detection ran, so `duplicates` and `exact_families` are empty and `traditional`/`semantic` are `null` in `run`).

`potentially_unused` is ranked and bounded too: ids are ordered by line span (`end_line - line + 1`) descending, then statement count, then file position, so the largest dead definitions lead, and the list holds at most 20 (`--max-unused`, recorded as `summary.max_unused`; `--max-unused all` removes the cap and the expansion flags above lift it unless an explicit value is given). `summary.potentially_unused` stays the complete count while `summary.reported_unused` and `summary.truncated_unused` split it into emitted and cut; units referenced only by cut unused findings leave `units`. `summary.suppressed_unused` counts units that carry a `codedupes: ignore`/`codedupes: ignore[unused]` directive and would otherwise have been a finding; a directive on a unit that was already exempt some other way (public surface, `get_*`/`set_*`, a test file) is not counted. `extraction_diagnostics`, `semantic_diagnostics`, `unused_diagnostics`, and the raw `--show-all` edge lists are deliberately complete: they are per-file records a consumer needs in full, so a scan with many diagnostics still produces a large document.

See [hybrid confidence tiers](analysis-defaults.md#hybrid-synthesis-confidence-defaults) to interpret `tier` and `score`.

#### Focused reports

`--focus PATH` (repeatable) scopes the emitted report and the exit code to findings touching one or more paths, without changing what the analyzer scanned: `units`, `run`, and every diagnostics array stay corpus-wide, so the underlying scan is identical with or without `--focus`. Only `duplicates`, `exact_families`, `potentially_unused`, and the counts derived from them shrink to the focus scope; `total_units` and `units` still reflect the full corpus, minus units referenced only by findings the focus dropped. Each focus path must exist inside the scan root; an in-tree file symlink that extraction follows keeps its alias even when its target is outside the root. The target itself must be a directory — a file target with `--focus` is a usage error naming the root to scan instead.

`summary.focus` is `null` for an unfocused report and otherwise `{"paths": [...], "units": N, "out_of_focus_duplicates": N, "out_of_focus_unused": N}`: `paths` is the resolved, sorted focus paths as strings, `units` counts corpus units under those paths, and the two `out_of_focus_*` counts are what focusing removed, in the same finding units as `summary.hybrid_duplicates`/`summary.potentially_unused` (a family counts once). An exact-duplicate family is kept whole when any of its members is in focus — consolidating it is one indivisible finding, so a focused report never shows half a family — while every other duplicate pair is kept when either endpoint is in focus, and a potentially-unused unit is kept when its file is in focus. `select_findings(complete).total_findings == select_findings(focused).total_findings + focus.out_of_focus_duplicates` always holds, so the two runs' counts reconcile.

The exit code is computed on the focused findings, so `--focus` narrows what `--fail-on` can fail on; a run that fails on out-of-focus findings can pass when focused. `--fail-on-incomplete` still applies to the corpus-wide `analysis_status`, including under `--focus`, because focusing does not change which files were analyzed or which diagnostics were recorded. `--focus` composes with `--unused-only` and the other detection modes: it filters whichever finding lists that mode produces.

### Search

Default search hits (`--result-level unit`) use `{"unit": "u0", "score": 0.95}`; their unit records have the same fields as check results. An empty index with `--no-cache` produces:

```json
{
  "schema_version": 4,
  "query": "refund validation",
  "analysis_status": "empty",
  "run": {
    "tool_version": "0.9.0",
    "root": "/repo/empty",
    "target": "/repo/empty",
    "languages": null,
    "exclude_patterns": [],
    "respect_gitignore": true,
    "include_private": true,
    "include_stubs": false,
    "extracted_files": 0,
    "units": {
      "extracted": 0,
      "semantic_eligible": 0,
      "by_language": {},
      "by_type": {"class": 0, "function": 0, "method": 0}
    },
    "traditional": null,
    "semantic": {
      "requested_model": "Alibaba-NLP/gte-modernbert-base",
      "model": "Alibaba-NLP/gte-modernbert-base",
      "revision": null,
      "profile": "gte-modernbert",
      "threshold_profile": "auto",
      "task": "code-search-query",
      "device": "auto",
      "execution_device": null,
      "thresholds": {},
      "threshold_floor": 0.0,
      "min_statements": 3,
      "unit_types": ["function", "method"],
      "cross_language": false,
      "hybrid_split": null
    },
    "unused": null,
    "checks": {
      "extraction": {"status": "empty", "files": 0, "files_failed": 0, "diagnostics": 0},
      "traditional": {"status": "disabled", "files": null, "files_failed": 0, "diagnostics": 0},
      "semantic": {"status": "completed", "files": null, "files_failed": 0, "diagnostics": 0},
      "unused": {"status": "disabled", "files": null, "files_failed": 0, "diagnostics": 0}
    }
  },
  "summary": {
    "indexed_units": 0,
    "extracted_units": 0,
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
    },
    "query_execution": []
  },
  "results": [],
  "units": {},
  "extraction_diagnostics": [],
  "semantic_diagnostics": []
}
```

`summary.indexed_units` is the semantic corpus size after eligibility filtering; `summary.extracted_units` (`run.units.extracted`) is the pre-filter extraction count, so the two distinguish an empty repository from a populated one that eligibility filtering emptied. `summary.query_execution` lists one `{"execution_device", "cache_hit"}` record per query vector this search resolved (empty for `--result-level file`'s per-match grouping, which does not issue extra queries); `cache_hit: true` and `execution_device: null` together mean the query embedding came from the persistent cache without loading the model.

An empty index means one of three different things, and `analysis_status`/`run.checks.extraction` distinguish them:

- **Empty repository**: extraction itself produced no units. `run.units.extracted` is `0`, `run.checks.extraction.status` is `"empty"`, and `analysis_status` is `"empty"` (the example above). The terminal warns that extraction produced no code units.
- **Filtered to empty**: extraction produced units, but semantic eligibility filtering (`--min-statements`, `--semantic-unit-type`) removed every one of them. `run.units.extracted` is nonzero while `summary.indexed_units` is `0`, `run.checks.extraction.status` is `"completed"`, and `analysis_status` is `"complete"`. The terminal warns that candidate filtering emptied the index and names the filters to loosen.
- **Populated but no matches**: the index has units and the query ran, but nothing scored above the search threshold. `summary.indexed_units` is nonzero, `results` is `[]`, and `analysis_status` is `"complete"`.

An empty terminal index warns on stderr and distinguishes empty extraction from eligibility filtering the same way.

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

`check` emits `extraction_diagnostics`, `semantic_diagnostics`, and `unused_diagnostics` arrays; `search` emits the extraction and semantic diagnostic arrays (search runs no unused analysis, so it has no `unused_diagnostics`). All three arrays stay complete regardless of report caps or `run.checks`, so recoverable failures remain visible even when the search index is empty. Entries use `file`, `language`, `severity`, `code`, `message`, `line`, and `end_line`. `unused_diagnostics` codes are `unused-read-error`, `unused-parse-error`, and `unused-recursion-limit`, one entry per file the unused reference walk could not process.

Counts moved off `summary` and onto [`run.checks`](#run-record-and-check-status): `run.checks.extraction.diagnostics` is `len(extraction_diagnostics)` and `run.checks.extraction.files_failed` counts the distinct files behind a scope-losing diagnostic (a `partial-parse`/`read-error`/etc., not an advisory notice like `c-header-policy`); `run.checks.semantic.diagnostics` is `len(semantic_diagnostics)`; `run.checks.unused.diagnostics`/`files_failed` mirror `len(unused_diagnostics)` for `check`. Both terminal commands still print up to ten entries per diagnostic category; `check`'s `Run` panel and summary show the derived check statuses instead of raw counts.

For `semantic-context-overflow` warnings and their cache behavior, see [long-input handling](analysis-defaults.md#semantic-candidate-defaults).

## Exit codes

`check --fail-on` controls findings only; runtime and usage failures retain their normal status:

- `--fail-on actionable` (default): combined mode exits `1` for `exact`, `traditional_near`, or `hybrid_confirmed`. Pure-semantic `semantic_high_confidence` pairs are reported but advisory, and `semantic_review` pairs are withheld and advisory, because neither has deterministic structural/token corroboration. Non-strict unused guesses are also advisory, while `--strict-unused` makes them actionable. Raw single-method duplicates already passed the explicitly selected method thresholds and remain actionable.
- `--fail-on all`: any duplicate or unused finding in the complete result exits `1`, including `semantic_review` pairs the report withheld.
- `--fail-on none`: findings never change the successful exit code.

The exit code is computed on the complete analysis result before report selection, so `--include-review`, `--show-all`, `--max-duplicates`, and `--max-unused` never change it. `--focus` is the one flag that does change it, since it is applied before this computation; see [focused reports](#focused-reports). The only way every failing finding can be hidden from the report is `--fail-on all` with withheld `semantic_review` pairs; the terminal `Finding status` row then says so and points at `--include-review`, and JSON records `"hidden_only_failure": ["review"]` (otherwise `[]`). The report caps cannot cause this: families and actionable tiers rank first and unused failure is all-or-nothing, so whenever a cut finding fails, an emitted finding fails too. The selected policy, the unused strictness it was evaluated with, and the computed result are always present as `summary.fail_on`, `summary.strict_unused`, and `summary.exit_code`. Terminal summaries show the same values as `Failure policy` and `Finding status` rows.

`check --fail-on-incomplete` adds a second, independent failure source on top of `--fail-on`: it exits `1` whenever [`analysis_status`](#run-record-and-check-status) is not `"complete"` — empty extraction, a file-level extraction failure, a combined-mode semantic fallback, or a file the unused walk had to skip — even under `--fail-on none`, and even when the run found no findings at all. It never changes `--fail-on`'s own verdict; the two are ORed together. `summary.fail_on_incomplete` records whether it was set. The terminal `Finding status` row appends `analysis {status}: {reasons} fails --fail-on-incomplete` inside the existing `fail (exit 1; …)` form when it is the reason (or a contributing reason) for the failure; a run that fails only because of ordinary findings still reads `fail (exit 1)` verbatim. Fallback counts as partial even with `--allow-semantic-fallback`, since the flag's purpose is to keep the run from being fatal, not to declare the degraded result complete.

Command status conventions:

- `0`: command completed and the selected finding policy (and `--fail-on-incomplete`, if set) did not fail the run.
- `1`: selected findings failed `check`, or `--fail-on-incomplete` was set and the analysis did not complete.
- `2`: CLI usage or validation error.
- `3`: runtime failure — the command did not complete (parser unavailable, an unhandled backend exception, a path that disappeared mid-run, or a cache operation that failed outright). `cache info`/`cache clear` use the same code for their own runtime failures.

Default combined semantic backend or runtime failures are fatal (exit `3`). `--allow-semantic-fallback` continues with full-scope traditional results and records `summary.semantic_fallback` plus `summary.semantic_fallback_reason`; under the default actionable policy, heuristic unused findings alone do not turn that successful degraded run into exit `1` — add `--fail-on-incomplete` to fail on the degradation itself.

## Terminal duplicate panels

`check` prints a `Run` panel before anything else: tool version, `Root`/`Target`, `Scope` (the resolved `analysis_mode`), a `Checks` line (`extraction=completed, traditional=completed, semantic=completed, unused=completed`, one entry per detector), then a `Traditional`/`Semantic` line for each detector that ran, and `Units` (extracted vs. semantic-eligible counts). It never lists findings — the summary and finding panels below it do — so a reader learns what was configured and what completed before seeing what it found. The `Analysis Summary` table gains an `Analysis status` row showing the same `complete`/`partial`/`empty` value as JSON's `analysis_status`. Under `--focus`, three more rows appear right after it — `Focus` (the resolved paths), `Out-of-focus duplicates`, and `Out-of-focus unused` — mirroring `summary.focus`; every count below them in the table is already scoped to the focus.

The primary panels list every finding the [report caps](#report-selection) selected, so they show exactly what `--json` would emit. `Exact Duplicate Families (N families, K truncated)` comes first when any family is kept, one row per family with member count, `Lines`, `Method`, the first member, and up to three more locations before a `+N more` note; `--show-source` prints one snippet per member, bounded to `--source-lines` lines (default `40`) with a trailing `... (N more lines)` note when cut. `--show-diff` adds a unified diff per row after any source panels, also bounded by `--source-lines` with a trailing `... (N more diff lines)` note: pair tables diff both units, `structural_hash` families diff each member against the first, and `token_hash` families print nothing extra because their members are already token-for-token identical. The pair table follows, and the unused table (`Potentially Unused (N units, K truncated)`, same title in every mode) lists the largest units first with a `Lines` column and a blurb stating which units are excluded (`--strict-unused` reports public functions and methods too). With `--show-source`, that table is followed by one bounded source panel per unused unit, including in unused-only mode. Only the raw `--show-all` tables keep a 20-row display limit; their footers point to `--full-table`, which lifts that limit and, unless given explicitly, the `--max-duplicates` and `--max-unused` caps as well.

Every table names units by `qualified_name`, not the bare name, so a shared function name is distinguishable across files and nesting; module prefixes are never elided. Locations use the shorter of working-directory-relative and absolute `<path>:<line>` spellings.

- Combined: `Hybrid Duplicates (N pairs, M review withheld, K truncated)`, followed by any raw panels requested through [report selection](#report-selection). When every hybrid pair is withheld, one dim line reports the withheld count instead of an empty table. The summary lists every tier's count, the `exact` row reading `N families (M units)`, plus `Actionable duplicates` as `total (reported)`; withheld and truncated totals appear when non-zero, the latter naming the cut tiers (families as `N exact families`) and pointing at `--max-duplicates all`, and `Truncated unused` points at `--max-unused all`; `Unused policy` reads `strict` or `default` to match `--strict-unused`.
- `--traditional-only`: `Near Duplicates (Jaccard)` for the non-exact pairs (exact pairs are already in the family panel), after the family panel; the summary adds an `Exact duplicate families` row.
- `--semantic-only`: `Semantic Duplicates (Embedding)`.
- `--unused-only`: no duplicate panels or families print at all (`run.traditional`/`run.semantic` are `null`); only the summary's unit counts and `Potentially Unused` table appear. `--fail-on actionable` (the default) never fails on its own, since non-strict unused findings are advisory; add `--strict-unused` or `--fail-on all` to fail on what it finds.
