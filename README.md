# codedupes

`codedupes` detects duplicate and potentially unused Python code with:

- Traditional AST/token matching (exact + Jaccard near-duplicate)
- Semantic matching with model-profile embeddings (default `gte-modernbert-base`)
- Heuristic unused-code detection

## Install

```bash
pip install "codedupes @ git+https://github.com/pszemraj/codedupes.git"
```

Optional GPU extras:

```bash
pip install "codedupes[gpu] @ git+https://github.com/pszemraj/codedupes.git"
```

Requires Python 3.11+. Details are in
[docs/install.md](https://github.com/pszemraj/codedupes/blob/main/docs/install.md)

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

## Documentation

Primary docs live under `docs/`:

- [docs/index.md](https://github.com/pszemraj/codedupes/blob/main/docs/index.md): documentation map and ownership
- [docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md): commands, flags, and defaults
- [docs/output.md](https://github.com/pszemraj/codedupes/blob/main/docs/output.md): JSON schemas and exit codes
- [docs/usage.md](https://github.com/pszemraj/codedupes/blob/main/docs/usage.md): practical workflows and tuning examples
- [docs/python-api.md](https://github.com/pszemraj/codedupes/blob/main/docs/python-api.md): programmatic API usage
- [docs/hybrid-tuning.md](https://github.com/pszemraj/codedupes/blob/main/docs/hybrid-tuning.md): hybrid gate tuning workflow

### Default semantic model behavior

See [docs/install.md](https://github.com/pszemraj/codedupes/blob/main/docs/install.md).

## Notes and limits

- Call graph and unused detection are heuristic and conservative by default.
- Semantic duplicate defaults embed function/method code units; include classes with `--semantic-unit-type class`.
- Traditional duplicate defaults suppress tiny wrapper noise (tiny exact pairs removed, tiny near pairs require high Jaccard).
- Semantic analysis may download model weights on first use.
- Extraction skips common artifact/cache directories by default (`__pycache__`, `.venv`, etc).
