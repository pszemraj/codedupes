# codedupes

`codedupes` detects duplicate and potentially unused Python code with:

- Traditional AST/token matching (exact + Jaccard near-duplicate)
- Semantic matching with C2LLM embeddings (`codefuse-ai/C2LLM-0.5B`)
- Heuristic unused-code detection

## Install

Source-of-truth install and runtime defaults:
[docs/install.md](https://github.com/pszemraj/codedupes/blob/main/docs/install.md).

## Quick Start

```bash
codedupes check ./src
codedupes search ./src "normalize request payload"
codedupes info
```

`codedupes check` defaults to a hybrid-first report:

- one combined duplicate list (`Hybrid Duplicates`)
- likely dead code (`potentially_unused`)

Use `--show-all` to include raw traditional + raw semantic duplicate lists.

## Default semantic model behavior

See [docs/install.md](https://github.com/pszemraj/codedupes/blob/main/docs/install.md).

## Documentation

Primary docs live under `docs/`:

- [docs/index.md](https://github.com/pszemraj/codedupes/blob/main/docs/index.md): documentation map and ownership
- [docs/install.md](https://github.com/pszemraj/codedupes/blob/main/docs/install.md): install and runtime defaults (source of truth)
- [docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md): commands, flags, and defaults (source of truth)
- [docs/output.md](https://github.com/pszemraj/codedupes/blob/main/docs/output.md): JSON schemas and exit codes (source of truth)
- [docs/usage.md](https://github.com/pszemraj/codedupes/blob/main/docs/usage.md): practical workflows and tuning examples
- [docs/python-api.md](https://github.com/pszemraj/codedupes/blob/main/docs/python-api.md): programmatic API usage
- [docs/hybrid-tuning.md](https://github.com/pszemraj/codedupes/blob/main/docs/hybrid-tuning.md): hybrid gate tuning workflow

## Notes and limits

- Call graph and unused detection are heuristic and conservative by default.
- Semantic analysis may download model weights on first use.
- Extraction skips common artifact/cache directories by default (for example `target`,
  `node_modules`, `__pycache__`, `.venv`, `build`, and `dist`).
