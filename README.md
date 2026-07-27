# codedupes

`codedupes` detects duplicate and potentially unused Python code with:

- Traditional AST/token matching (exact + Jaccard near-duplicate)
- Semantic matching with model-profile embeddings (default `gte-modernbert-base`)
- Persistent, content-addressed on-disk embedding cache so unchanged code never re-embeds
- Explicit CPU, CUDA, and Apple Silicon MPS execution
- Heuristic unused-code detection

## Install

```bash
pip install "codedupes @ git+https://github.com/pszemraj/codedupes.git"
```

Optional GPU extras:

```bash
pip install "codedupes[gpu] @ git+https://github.com/pszemraj/codedupes.git"
```

Requires Python 3.11+ and PyTorch 2.13+. Details are in
[docs/install.md](https://github.com/pszemraj/codedupes/blob/main/docs/install.md)

## Quick Start

```bash
codedupes check ./src
codedupes search ./src "normalize request payload"
codedupes info

# Apple Silicon
codedupes check ./src --device mps
```

`codedupes check` defaults to a hybrid-first report:

- one combined duplicate list (`Hybrid Duplicates`)
- likely dead code (`potentially_unused`)

Use `--show-all` to include raw traditional + raw semantic duplicate lists.

## Documentation

Primary docs live under `docs/`:

- [docs/index.md](https://github.com/pszemraj/codedupes/blob/main/docs/index.md): documentation map and ownership
- [docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md): commands, flags, and defaults
- [docs/caching.md](https://github.com/pszemraj/codedupes/blob/main/docs/caching.md): persistent embedding cache design, env vars, and `cache` subcommand
- [docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md): semantic model aliases, profile defaults, and task behavior
- [docs/accelerators.md](https://github.com/pszemraj/codedupes/blob/main/docs/accelerators.md): CPU/CUDA/MPS selection, Metal memory recovery, and MLX coexistence
- [docs/analysis-defaults.md](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md): analysis-behavior defaults and heuristics
- [docs/output.md](https://github.com/pszemraj/codedupes/blob/main/docs/output.md): JSON schemas and exit codes
- [docs/usage.md](https://github.com/pszemraj/codedupes/blob/main/docs/usage.md): practical workflows and tuning examples
- [docs/python-api.md](https://github.com/pszemraj/codedupes/blob/main/docs/python-api.md): programmatic API usage
- [docs/hybrid-tuning.md](https://github.com/pszemraj/codedupes/blob/main/docs/hybrid-tuning.md): hybrid gate tuning workflow

## Notes and limits

- Call graph and unused detection are heuristic and conservative by default.
- Semantic model-profile defaults and task behavior are defined in
  [docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md).
- Analysis defaults (semantic candidate scope, tiny-traditional filtering, hybrid gates) are defined in
  [docs/analysis-defaults.md](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md).
- Semantic analysis may download model weights on first use.
- Accelerator behavior and Apple Silicon guidance are defined in
  [docs/accelerators.md](https://github.com/pszemraj/codedupes/blob/main/docs/accelerators.md).
- Extraction skips common artifact/cache directories by default (`__pycache__`, `.venv`, etc).
