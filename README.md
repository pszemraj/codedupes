# codedupes

`codedupes` detects duplicate and potentially unused Python code with:

- Traditional AST/token matching (exact + Jaccard near-duplicate)
- Semantic matching with C2LLM embeddings (`codefuse-ai/C2LLM-0.5B`)
- Heuristic unused-code detection

## Install

```bash
pip install codedupes
```

Optional GPU extras:

```bash
pip install codedupes[gpu]
```

Requires Python 3.11+.

## Quick Start

```bash
codedupes check ./src
codedupes search ./src "normalize request payload"
codedupes info
```

## Documentation

Primary docs live under `docs/`:

- `docs/index.md`: documentation map
- `docs/usage.md`: practical workflows and tuning examples
- `docs/cli.md`: authoritative CLI command/option reference (source of truth)
- `docs/output.md`: authoritative JSON schema and exit codes (source of truth)
- `docs/python-api.md`: programmatic API usage

## Notes and limits

- Call graph and unused detection are heuristic and conservative by default.
- Semantic analysis may download model weights on first use.
- Extraction skips common artifact/cache directories by default (for example `target`,
  `node_modules`, `__pycache__`, `.venv`, `build`, and `dist`).
