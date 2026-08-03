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

See [Installation](https://github.com/pszemraj/codedupes/blob/main/docs/install.md) for Python and runtime requirements.

## Quick Start

```bash
codedupes check ./src
codedupes search ./src "normalize request payload"
codedupes info

# Apple Silicon
codedupes check ./src --device mps
```

See [Output and exit codes](https://github.com/pszemraj/codedupes/blob/main/docs/output.md) for report modes and CI behavior.

## Documentation

- [Installation](https://github.com/pszemraj/codedupes/blob/main/docs/install.md)
- [CLI reference](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md)
- [Usage guide](https://github.com/pszemraj/codedupes/blob/main/docs/usage.md)
- [Python API](https://github.com/pszemraj/codedupes/blob/main/docs/python-api.md)
- [Output and exit codes](https://github.com/pszemraj/codedupes/blob/main/docs/output.md)
- [Analysis defaults and heuristics](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md)
- [Semantic model profiles and tasks](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md)
- [Embedding cache](https://github.com/pszemraj/codedupes/blob/main/docs/caching.md)
- [Accelerators and Apple Silicon](https://github.com/pszemraj/codedupes/blob/main/docs/accelerators.md)
- [Hybrid gate tuning](https://github.com/pszemraj/codedupes/blob/main/docs/hybrid-tuning.md)

## Notes and limits

- Call graph and unused detection are heuristic and conservative by default.
- Semantic analysis may download model weights on first use.
- Extraction skips common artifact/cache directories by default (`__pycache__`, `.venv`, etc).
