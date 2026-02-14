# codedupes Documentation

`codedupes` detects duplicate and potentially unused Python code using:

- Traditional AST and token-based duplicate detection
- Near-duplicate matching with Jaccard similarity
- Semantic matching with embedding similarity
- Heuristic unused-code detection

## Documentation Map

- `docs/usage.md`: install, quick start, and common workflows
- `docs/cli.md`: full CLI command and option reference (source of truth for flags/defaults)
- `docs/output.md`: output schema and exit codes (source of truth)
- `docs/python-api.md`: programmatic API usage

## Quick Start

```bash
pip install codedupes
codedupes check ./src
```

For JSON output, schema details are in `docs/output.md`:

```bash
codedupes check ./src --json
```
