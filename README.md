# codedupes

`codedupes` detects duplicate and potentially unused Python code with:

- Traditional AST/token matching (exact + Jaccard near-duplicate)
- Semantic matching with C2LLM embeddings (`codefuse-ai/C2LLM-0.5B`)
- Optional unused candidate detection

This document is the primary usage reference.

## Docs

Detailed documentation is available under `docs/`:

- `docs/index.md`
- `docs/usage.md`
- `docs/cli.md`
- `docs/python-api.md`
- `docs/output.md`

## Install

```bash
pip install codedupes
```

For GPU-assisted semantic models that may require `deepspeed`, install:

```bash
pip install codedupes[gpu]
```

Requires Python 3.10+.

## CLI usage

### `check`

```bash
codedupes check <path>
```

Run full duplicate + unused analysis.

```bash
codedupes check ./src
codedupes check ./src --json --threshold 0.82
codedupes check ./src --semantic-only --traditional-threshold 0.8
codedupes check ./src --traditional-only --no-unused
```

### `search`

```bash
codedupes search <path> "<query>"
```

Search for code using natural language query.

```bash
codedupes search ./src "sum values in a list" --top-k 5
codedupes search ./src "remove unused values" --json
```

### `info`

```bash
codedupes info
```

Prints version/model/config defaults.

### Compatibility mode (legacy invocation retained)

```bash
codedupes <path> [options]
```

is retained as a compatibility mode and is equivalent to:

```bash
codedupes check <path> [options]
```

## `check` options

- `-t, --threshold`
  - Shared threshold for both methods (default `0.82`).
- `--semantic-threshold`
  - Override semantic threshold only.
- `--traditional-threshold`
  - Override traditional (Jaccard) threshold only.
- `--semantic-only`
  - Run semantic only.
- `--traditional-only`
  - Run traditional only.
- `--no-unused`
  - Skip unused-code detection.
- `--strict-unused`
  - Do not skip public top-level functions from unused scanning.
- `--no-private`
  - Exclude private (`_prefixed`) functions/classes.
- `--min-lines`
  - Minimum statement count for semantic comparison (default `3`).
- `--model`
  - Embedding model override (default `codefuse-ai/C2LLM-0.5B`).
- `--model` + `--batch-size`
  - Adjust embedding throughput.
- `--json`
  - Emit machine-readable results.
- `--show-source`
  - Print duplicate snippets in terminal output.
- `--include-stubs`
  - Include `*.pyi` files for analysis.
- `--exclude`
  - Glob patterns to skip files.
- `-v, --verbose`
  - Enable verbose logs.

## `search` options

- `--top-k`
  - Number of results (default `10`).
- `--model`
- `--no-private`
- `--json`

## Analysis defaults

- Semantic threshold default: `0.82`
- Traditional threshold default: `0.85`
- Semantic min-lines default: `3` (trivial bodies are skipped from semantic pipeline)
- Unused detection default: enabled (`--no-unused` to disable)
- Public top-level functions are treated as API-like by default and are not flagged as unused unless `--strict-unused`.

## Programmatic API

```python
from codedupes import analyze_directory

result = analyze_directory("./src", semantic_threshold=0.82, traditional_threshold=0.85)
for dup in result.exact_duplicates:
    print(dup.unit_a.qualified_name, "≈", dup.unit_b.qualified_name)

for unit in result.potentially_unused:
    print("Unused:", unit.qualified_name)
```

Search programmatically:

```python
from codedupes import CodeAnalyzer, AnalyzerConfig

analyzer = CodeAnalyzer(AnalyzerConfig(min_semantic_lines=3))
analyzer.analyze("./src")
for unit, score in analyzer.search("load csv"):
    print(score, unit.qualified_name)
```

## Output interpretation

- `exact_duplicates`: AST/token detections (`ast_hash`, `token_hash`)
- `semantic_duplicates`: cosine threshold hits from embeddings
- `potentially_unused`: units with no detected references and not considered public API

## Notes and limits

- `C2LLM_INSTRUCTIONS["code"]` is tuned for code2code retrieval.
- Call graph is intra-project only.
- Unused detection remains heuristic and conservative; tune with `--strict-unused` if needed.
- For deterministic CI, mock `codedupes.semantic.get_model` or use small fake model doubles instead of downloading remote weights.
