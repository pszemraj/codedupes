# Output and Exit Codes

This document is the source of truth for machine-readable output and CLI exit semantics.

## `check --json` Structure

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
- Semantic backend note: in mixed-mode `check` (default), semantic failures degrade to
  traditional/unused-only results with a warning instead of hard failure.
- Combined-mode finding note: default `check` exits `1` based on hybrid findings
  (`hybrid_duplicates`) plus `potentially_unused` only.

`search`:

- `0`: completed successfully
- `1`: failed due to runtime error
- `2`: CLI usage/validation error (Click)
- Semantic backend note: `search` requires semantic inference and fails hard if semantic
  backend loading/inference fails.

`info`:

- `0`: completed successfully
