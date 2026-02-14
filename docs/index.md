# codedupes Documentation

`codedupes` detects duplicate and potentially unused Python code using:

- Traditional AST and token-based duplicate detection
- Near-duplicate matching with Jaccard similarity
- Semantic matching with embedding similarity
- Heuristic unused-code detection

## Documentation Map

- `docs/usage.md`: install, quick start, and common workflows
- `docs/cli.md`: full CLI command and option reference
- `docs/python-api.md`: programmatic API usage
- `docs/output.md`: output schema, exit codes, and interpretation

## Quick Start

```bash
pip install codedupes
codedupes check ./src
```

For JSON output:

```bash
codedupes check ./src --json
```
