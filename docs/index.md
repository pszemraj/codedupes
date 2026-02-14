# codedupes Documentation

`codedupes` detects duplicate and potentially unused Python code using:

- Traditional AST and token-based duplicate detection
- Near-duplicate matching with Jaccard similarity
- Semantic matching with embedding similarity
- Heuristic unused-code detection

## Documentation ownership (source-of-truth model)

- `docs/cli.md`: source of truth for CLI commands, flags, and defaults.
- `docs/output.md`: source of truth for JSON payload shapes and exit codes.
- `docs/usage.md`: workflows and tuning recipes. Links to CLI/output docs for definitive flag/schema semantics.
- `docs/python-api.md`: programmatic API usage and result objects.
- `docs/hybrid-tuning.md`: best-practice workflow for hybrid gate sweep and threshold updates.

## Quick Start

```bash
pip install codedupes
codedupes check ./src
```

For machine-readable output, use JSON:

```bash
codedupes check ./src --json
```

For full command/option details, see `docs/cli.md`.
