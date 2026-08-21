# Output and Exit Codes

## `check --json` schemas

`check` has two JSON schema modes:

1. Combined mode (default): hybrid-first output
2. Single-method mode (`--semantic-only` or `--traditional-only`): raw output

## Combined mode (default)

`codedupes check <path> --json` emits:

```json
{
  "analysis_mode": "combined",
  "summary": {
    "total_units": 0,
    "units_by_language": {},
    "hybrid_duplicates": 0,
    "potentially_unused": 0,
    "raw_traditional_duplicates": 0,
    "raw_semantic_duplicates": 0,
    "semantic_fallback": false,
    "semantic_fallback_reason": null,
    "extraction_diagnostics": 0,
    "semantic_diagnostics": 0,
    "unused_supported_languages": ["python"],
    "unused_excluded_units": 0
  },
  "extraction_diagnostics": [],
  "semantic_diagnostics": [],
  "hybrid_duplicates": [],
  "potentially_unused": []
}
```

With `--show-all`, additional raw sections are included:

- `traditional_duplicates`
- `semantic_duplicates`

## Single-method mode (`--semantic-only` or `--traditional-only`)

`codedupes check <path> --json --semantic-only` and `codedupes check <path> --json --traditional-only` emit raw duplicate sections:

```json
{
  "analysis_mode": "semantic",
  "summary": {
    "total_units": 0,
    "units_by_language": {},
    "traditional_duplicates": 0,
    "semantic_duplicates": 0,
    "potentially_unused": 0,
    "semantic_fallback": false,
    "semantic_fallback_reason": null,
    "extraction_diagnostics": 0,
    "semantic_diagnostics": 0,
    "unused_supported_languages": ["python"],
    "unused_excluded_units": 0
  },
  "extraction_diagnostics": [],
  "semantic_diagnostics": [],
  "traditional_duplicates": [],
  "semantic_duplicates": [],
  "potentially_unused": []
}
```

`hybrid_duplicates` is only part of default combined mode. `analysis_mode` is always present (`combined`, `traditional`, `semantic`, or `none`).

Each duplicate entry includes:

- `unit_a`
- `unit_b`

`hybrid_duplicates` entries include:

- `tier` (`exact`, `traditional_near`, `hybrid_confirmed`, `semantic_high_confidence`, or `semantic_review`)
- `confidence`
- evidence fields (`has_exact`, `semantic_similarity`, `jaccard_similarity`, etc.)

`semantic_review` means the pair cleared its calibrated semantic duplicate gate but lacks the lexical or statement-count corroboration used for the high-confidence tier. It remains visible because alpha-renaming and structural translation often remove exactly that lexical overlap, and its confidence is scaled to rank below every corroborated tier (see [Analysis defaults](analysis-defaults.md#confidence-scale)).

Raw duplicate entries include:

- `similarity`
- `method`

Each unit object includes:

- Identity: `name`, `qualified_name`, and `type`
- Language: `language`, `dialect`, and `native_kind`
- Location: `file`, `line`, `end_line`, `start_byte`, `end_byte`, `start_column`, and `end_column`
- Extraction metadata: `statement_count`
- Visibility: `is_public` and `is_exported`

## Diagnostics

`extraction_diagnostics` and `semantic_diagnostics` are separate top-level arrays with a matching count in `summary`. Both use the same entry shape: `file`, `language`, `severity`, `code`, `message`, `line`, and `end_line` (the last two are `null` for file-level diagnostics). The terminal summary prints a count row per non-empty list and shows the first ten entries of each.

`extraction_diagnostics` covers parsing and file selection:

- `parse-error`: a file the parser could not read at all
- `partial-parse`: Tree-sitter recovered from invalid or incomplete source; units whose own subtree contains an error are omitted with `unit-parse-error`
- `unit-parse-error`: one extracted unit skipped because its own syntax subtree contains an error
- `c-header-policy`: one summary diagnostic per run when `.h` files are skipped by the conservative C-header policy during a directory scan, naming the count and suggesting `--language c`
- `declaration-file`: an explicitly named `.d.ts`/`.d.mts`/`.d.cts` file, which has no implementation bodies
- `language-filter`: an explicitly named file excluded by `--language`
- `unsupported-file`: an explicitly named file no extraction backend accepts

The last three are raised only for files named on the command line. A directory scan silently passes over unsupported and filtered files; only the C-header summary is reported.

`semantic_diagnostics` covers corpus units the semantic stage dropped. The only current code is `semantic-context-overflow`: the unit's tokenized input, encode prompt included, exceeds the model's context window, so it is skipped instead of embedded from a partial prefix. The run continues without it. An over-long `search` query still fails hard.

## `search --json` Structure

`codedupes search <path> "<query>" --json` emits:

```json
{
  "query": "text",
  "results": [
    {
      "score": 0.95,
      "name": "func",
      "qualified_name": "pkg.mod.func",
      "type": "function",
      "language": "python",
      "dialect": "python",
      "native_kind": "FunctionDef",
      "file": "src/pkg/mod.py",
      "line": 10,
      "end_line": 20,
      "start_byte": 120,
      "end_byte": 480,
      "start_column": 0,
      "end_column": 17,
      "statement_count": 4,
      "is_public": true,
      "is_exported": false
    }
  ],
  "semantic_diagnostics": []
}
```

## Terminal duplicate panels

- combined mode: `Hybrid Duplicates`, plus `Traditional Duplicates (Raw Structural/Token/Jaccard)` and `Semantic Duplicates (Raw Embedding)` under `--show-all`
- `--traditional-only`: `Traditional Duplicates (Structural/Token/Jaccard)`
- `--semantic-only`: `Semantic Duplicates (Embedding)`

## Exit Codes

`check`:

- `0`: completed, no findings
- `1`: completed with findings or failed due to runtime error
- `2`: CLI usage/validation error (Click)
- Semantic backend note:
  - default combined `check`: semantic failures fail hard
  - `--allow-semantic-fallback`: combined mode can continue with scoped traditional results, and degraded runs are surfaced in JSON as `summary.semantic_fallback` plus `summary.semantic_fallback_reason`
  - semantic-required mode (`--semantic-only`): fails hard
- Finding note:
  - combined mode: exit `1` is based on `hybrid_duplicates` + `potentially_unused`
  - single-method mode: exit `1` is based on raw duplicate findings + `potentially_unused`

`search`:

- `0`: completed successfully
- `1`: failed due to runtime error
- `2`: CLI usage/validation error (Click)
- Semantic backend note: `search` requires semantic inference and fails hard if semantic backend loading/inference fails.

`info` and `cache info`:

- `0`: completed successfully

`cache clear`:

- `0`: cleared successfully
- `1`: failed to clear cache state
