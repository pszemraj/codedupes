# Output and Exit Codes

This document is the source of truth for machine-readable output and CLI exit semantics.

## `check --json` Structure

`codedupes check <path> --json` emits:

```json
{
  "summary": {
    "total_units": 0,
    "exact_duplicates": 0,
    "semantic_duplicates": 0,
    "potentially_unused": 0
  },
  "exact_duplicates": [],
  "semantic_duplicates": [],
  "potentially_unused": []
}
```

Each duplicate entry includes:

- `unit_a`
- `unit_b`
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

`search`:

- `0`: completed successfully
- `1`: failed due to runtime error
- `2`: CLI usage/validation error (Click)

`info`:

- `0`: completed successfully
