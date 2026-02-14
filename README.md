# codedupes

`codedupes` detects duplicate and potentially unused Python code units using two complementary
approaches:

- Traditional AST/token analysis for exact and near duplicates.
- Semantic embedding similarity for functionally similar code blocks.

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

## API

```python
from codedupes import analyze_directory

result = analyze_directory("./src", semantic_threshold=0.88, traditional_threshold=0.85)

for dup in result.exact_duplicates:
    print(dup.unit_a.qualified_name, "≈", dup.unit_b.qualified_name)

for unit in result.potentially_unused:
    print("Unused:", unit.qualified_name)
```

## CLI options

- `--threshold`
  - Shared similarity default for both methods.
- `--semantic-threshold`
  - Override only semantic similarity.
- `--traditional-threshold`
  - Override only Jaccard near-duplicate threshold.
- `--semantic-only`
- `--traditional-only`
- `--no-unused`
  - Skip potentially-unused detection.
- `--no-private`
  - Exclude `_prefixed` functions, methods, and classes.
- `--json`
  - Machine-readable output.
- `--show-source`
  - Show snippets for duplicate findings.

## Output interpretation

- `exact_duplicates`: AST/hash-based detections (`ast_hash`, `token_hash`).
- `semantic_duplicates`: cosine similarity of code embeddings.
- `potentially_unused`: units with no detected references and not likely public API.

## Notes

- `findings` are heuristics, especially unused detection.
- For CI or deterministic tests, prefer mocking `codedupes.semantic.get_model` so no external model
  download is required.
- Parse failures are ignored per file with warnings; they do not stop analysis.
