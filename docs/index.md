# codedupes Documentation

`codedupes` detects duplicate and potentially unused Python code using:

- Traditional AST and token-based duplicate detection
- Near-duplicate matching with Jaccard similarity
- Semantic matching with embedding similarity
- Heuristic unused-code detection

## Documentation ownership (source-of-truth model)

- [docs/install.md](https://github.com/pszemraj/codedupes/blob/main/docs/install.md): installation and dependency/runtime environment setup.
- [docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md): source of truth for semantic model aliases, model-profile defaults, and semantic task behavior.
- [docs/accelerators.md](https://github.com/pszemraj/codedupes/blob/main/docs/accelerators.md): source of truth for CPU/CUDA/MPS selection, PyTorch MPS recovery, precision policy, and MLX coexistence.
- [docs/analysis-defaults.md](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md): source of truth for analysis-behavior defaults and heuristics.
- [docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md): source of truth for CLI commands, flags, and defaults.
- [docs/caching.md](https://github.com/pszemraj/codedupes/blob/main/docs/caching.md): source of truth for the persistent embedding cache (location, env vars, `cache` subcommand).
- [docs/output.md](https://github.com/pszemraj/codedupes/blob/main/docs/output.md): source of truth for JSON payload shapes and exit codes.
- [docs/usage.md](https://github.com/pszemraj/codedupes/blob/main/docs/usage.md): workflows and tuning recipes. Links to CLI/output docs for definitive flag/schema semantics.
- [docs/python-api.md](https://github.com/pszemraj/codedupes/blob/main/docs/python-api.md): programmatic API usage and result objects.
- [docs/hybrid-tuning.md](https://github.com/pszemraj/codedupes/blob/main/docs/hybrid-tuning.md): best-practice workflow for hybrid gate sweep and threshold updates.
- [docs/release-notes-next.md](https://github.com/pszemraj/codedupes/blob/main/docs/release-notes-next.md): implementation and validation scope for the next release.

## Quick Start

Install the CLI:
[docs/install.md](https://github.com/pszemraj/codedupes/blob/main/docs/install.md)

Then run:

```bash
codedupes check ./src
```

For machine-readable output, use JSON:

```bash
codedupes check ./src --json
```

For full command/option details, see
[docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md).
