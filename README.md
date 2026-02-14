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

For local development from this repo (without `PYTHONPATH=src` hacks):

```bash
pip install -e ".[dev]"
codedupes info
```

## Quick Start

```bash
codedupes check ./src
codedupes search ./src "normalize request payload"
codedupes info
```

`codedupes check` now defaults to a hybrid-first duplicate report (single combined
duplicate list + likely dead code). Use `--show-all` to also inspect raw
traditional and semantic duplicate lists.

## Default Semantic Model Behavior

- Default embedding model: `codefuse-ai/C2LLM-0.5B`
- Default pinned revision: `bd6d0ddb29f0c9a3d0f14281aedc9f940bb8d67a`
- Default loading for this model uses `trust_remote_code=True`
- C2LLM prefers `bfloat16` on CUDA (when supported) and on CPU (no `fp16` fallback)
- Default semantic embedding batch size: `8` with automatic CUDA OOM batch backoff

This means model-provided Python code from Hugging Face is executed when semantic
analysis is enabled. You can override both revision and trust behavior from the CLI:

```bash
codedupes check ./src --model-revision <commit_or_tag>
codedupes check ./src --no-trust-remote-code
codedupes check ./src --instruction-prefix "Represent this code for duplicate detection: "
```

## Troubleshooting Semantic Backend Issues

If semantic loading/inference fails, `codedupes check` falls back to traditional analysis
unless semantic-only mode is required. For compatibility in fresh environments, use:

```bash
pip install "transformers>=4.51,<5" "sentence-transformers>=5,<6"
```

Fallback-free baseline:

```bash
codedupes check ./src --traditional-only
```

Raw evidence debugging:

```bash
codedupes check ./src --show-all
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
