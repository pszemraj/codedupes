# Output and Exit Codes

This document is the source of truth for machine-readable output and CLI exit semantics.

## `check --json` schemas

`check` has two JSON schema modes:

1. Combined mode (default): hybrid-first output
2. Single-method mode (`--semantic-only` or `--traditional-only`): raw output

## Combined mode (default)

`codedupes check <path> --json` emits:

```json
{
  "summary": {
    "total_units": 0,
    "hybrid_duplicates": 0,
    "potentially_unused": 0,
    "raw_traditional_duplicates": 0,
    "raw_semantic_duplicates": 0,
    "filtered_raw_duplicates": 0
  },
  "hybrid_duplicates": [],
  "potentially_unused": []
}
```

With `--show-all`, additional raw sections are included:

- `traditional_duplicates`
- `semantic_duplicates`

## Single-method mode (`--semantic-only` or `--traditional-only`)

`codedupes check <path> --json --semantic-only` and
`codedupes check <path> --json --traditional-only` emit raw duplicate sections:

```json
{
  "summary": {
    "total_units": 0,
    "traditional_duplicates": 0,
    "semantic_duplicates": 0,
    "potentially_unused": 0
  },
  "traditional_duplicates": [],
  "semantic_duplicates": [],
  "potentially_unused": []
}
```

`hybrid_duplicates` is only part of default combined mode.

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

- `name`
- `qualified_name`
- `type`
- `file`
- `line`
- `end_line`
- `is_public`
- `is_exported`

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
  - default combined `check`: semantic failures degrade with a warning
  - semantic-required mode (`--semantic-only`): fails hard
- Finding note:
  - combined mode: exit `1` is based on `hybrid_duplicates` + `potentially_unused`
  - single-method mode: exit `1` is based on raw duplicate findings + `potentially_unused`

`search`:

- `0`: completed successfully
- `1`: failed due to runtime error
- `2`: CLI usage/validation error (Click)
- Semantic backend note: `search` requires semantic inference and fails hard if semantic
  backend loading/inference fails.

`info`:

- `0`: completed successfully
