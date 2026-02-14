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

`codedupes check` defaults to a hybrid-first report:

- one combined duplicate list (`Hybrid Duplicates`)
- likely dead code (`potentially_unused`)

Use `--show-all` to include raw traditional + raw semantic duplicate lists.

## Default semantic model behavior

- Model: `codefuse-ai/C2LLM-0.5B`
- Pinned revision: `bd6d0ddb29f0c9a3d0f14281aedc9f940bb8d67a`
- Default trust mode for this model: `trust_remote_code=True`
- Runtime dtype policy: prefer `bfloat16` (CUDA when supported, and CPU)
- Default semantic batch size: `8` with CUDA OOM backoff

Override trust/revision when needed:

```bash
codedupes check ./src --model-revision <commit_or_tag>
codedupes check ./src --no-trust-remote-code
```

If semantic loading/inference fails:

- default `check` degrades to traditional + unused analysis with a warning
- `search` and `check --semantic-only` fail hard

Recommended semantic dependency bounds:

```bash
pip install "transformers>=4.51,<5" "sentence-transformers>=5,<6"
```

Baseline without semantic backend:

```bash
codedupes check ./src --traditional-only
```

Debug raw evidence:

```bash
codedupes check ./src --show-all
```

## Documentation

Primary docs live under `docs/`:

- `docs/index.md`: documentation map and ownership
- `docs/cli.md`: source of truth for commands, flags, and defaults
- `docs/output.md`: source of truth for JSON schemas and exit codes
- `docs/usage.md`: practical workflows and tuning examples
- `docs/python-api.md`: programmatic API usage

## Notes and limits

- Call graph and unused detection are heuristic and conservative by default.
- Semantic analysis may download model weights on first use.
- Extraction skips common artifact/cache directories by default (for example `target`,
  `node_modules`, `__pycache__`, `.venv`, `build`, and `dist`).
