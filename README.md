# codedupes

`codedupes` detects duplicate and potentially unused Python code units.

This `README.md` is the **primary reference** for CLI usage, defaults, and output format.

Detection modes:

- Traditional AST/token analysis for exact and near-duplicate matching.
- Semantic embedding similarity for functional similarity.

## Primary behavior

- By default, both traditional and semantic analyses run.
- Potentially-unused candidates are reported unless `--no-unused` is passed.
- Parse errors are reported as warnings and do not fail the run.

## Install

```bash
pip install codedupes
```

Requires Python 3.10+.

## Quick usage

```bash
codedupes ./src
codedupes ./src --json
codedupes ./src --semantic-only --threshold 0.9
codedupes ./src --traditional-only --no-unused
```

## CLI usage reference

```bash
codedupes <path> [options]
```

Supported options:

- `-t, --threshold`
  - shared threshold for both similarity methods (default `0.85`).
- `--semantic-threshold`
  - semantic-only override.
- `--traditional-threshold`
  - Jaccard-only override.
- `--semantic-only`
  - run only semantic analysis.
- `--traditional-only`
  - run only AST/token analysis.
- `--no-unused`
  - skip unused-code reporting.
- `--no-private`
  - omit underscore-prefixed code units.
- `--model`
  - Hugging Face embedding model for semantic mode.
- `--json`
  - emit machine-readable JSON.
- `--show-source`
  - print duplicate source snippets.
- `-v, --verbose`
  - verbose logs.

## API

```python
from codedupes import analyze_directory

result = analyze_directory("./src", semantic_threshold=0.88, traditional_threshold=0.85)

for dup in result.exact_duplicates:
    print(dup.unit_a.qualified_name, "≈", dup.unit_b.qualified_name)

for unit in result.potentially_unused:
    print("Unused:", unit.qualified_name)
```

## Output interpretation

- `exact_duplicates`: AST/hash-based detections (`ast_hash`, `token_hash`).
- `semantic_duplicates`: cosine similarity of code embeddings.
- `potentially_unused`: units with no detected references and not likely public API.
- Findings are heuristic and should be manually reviewed.

## Configuration notes

- For deterministic CI tests, mock semantic model loading (`codedupes.semantic.get_model`) to avoid downloading remote weights.
- Exit status is `0` when no findings are detected, `1` when any exact/semantic duplicate or unused unit is found, and non-zero on errors.
