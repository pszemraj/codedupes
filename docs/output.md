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
    "filtered_raw_duplicates": 0,
    "semantic_fallback": false,
    "semantic_fallback_reason": null,
    "extraction_diagnostics": 0,
    "unused_supported_languages": ["python"],
    "unused_excluded_units": 0
  },
  "extraction_diagnostics": [],
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
    "unused_supported_languages": ["python"],
    "unused_excluded_units": 0
  },
  "extraction_diagnostics": [],
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

- `tier`
- `confidence`
- evidence fields (`has_exact`, `semantic_similarity`, `jaccard_similarity`, etc.)

Raw duplicate entries include:

- `similarity`
- `method`

Each unit object includes:

- Identity: `name`, `qualified_name`, and `type`
- Language: `language`, `dialect`, and `native_kind`
- Location: `file`, `line`, `end_line`, `start_byte`, `end_byte`, `start_column`, and `end_column`
- Extraction metadata: `has_body` and `statement_count`
- Visibility: `is_public` and `is_exported`

`extraction_diagnostics` contains file/language, severity, code, message, and optional line range. The terminal summary shows the first ten diagnostics. A `partial-parse` diagnostic means Tree-sitter recovered from invalid or incomplete source; units whose own subtree contains an error are omitted with `unit-parse-error`.

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
      "is_public": true,
      "is_exported": false
    }
  ]
}
```

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
