# Installation and Runtime Defaults

This page is the source of truth for installation, dependency bounds, and semantic runtime defaults.

## Install (GitHub source)

```bash
pip install "codedupes @ git+https://github.com/pszemraj/codedupes.git"
```

Optional GPU extras:

```bash
pip install "codedupes[gpu] @ git+https://github.com/pszemraj/codedupes.git"
```

Requires Python 3.11+.

## Local development (editable install)

```bash
git clone https://github.com/pszemraj/codedupes.git
cd codedupes
pip install -e ".[dev]"
codedupes info
```

## Semantic dependency bounds (default model)

```bash
pip install "transformers>=4.51,<5" "sentence-transformers>=5,<6"
```

## Default semantic runtime behavior

- Model: `codefuse-ai/C2LLM-0.5B`
- Default model revision: `auto` (resolves to `bd6d0ddb29f0c9a3d0f14281aedc9f940bb8d67a` for the default model)
- Default trust mode for this model: `trust_remote_code=True`
- Runtime dtype policy: prefer `bfloat16` (CUDA when supported, and CPU)
- Default semantic batch size: `8` with CUDA OOM backoff
- Failure behavior:
  - default `check` degrades to traditional + unused analysis with a warning
  - `search` and `check --semantic-only` fail hard

For CLI flags (including `--model-revision` and `--trust-remote-code`), see
[docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md).
