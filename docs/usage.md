# Usage Guide

This guide focuses on practical workflows.
Flag defaults and validation rules are defined in `docs/cli.md`.
JSON schema and exit codes are defined in `docs/output.md`.

## Install

```bash
pip install codedupes
```

Optional GPU extras:

```bash
pip install codedupes[gpu]
```

Recommended semantic dependency bounds:

```bash
pip install "transformers>=4.51,<5" "sentence-transformers>=5,<6"
```

## Core Workflow

1. Run analysis on a package or file.
2. Review the hybrid duplicate candidates.
3. Review potentially unused symbols.
4. Re-run with stricter/looser thresholds to tune sensitivity.

## Analyze A Project

```bash
codedupes check ./src
```

Inspect raw traditional + semantic evidence alongside hybrid output:

```bash
codedupes check ./src --show-all
```

Analyze one file:

```bash
codedupes check ./src/module.py
```

## Use JSON For CI/Automation

```bash
codedupes check ./src --json
```

See `docs/output.md` for authoritative JSON structure and exit-code semantics.

## Control Rich Output Width

Use a wider output width for less wrapping in terminal tables:

```bash
codedupes check ./src --output-width 200
```

## Search Semantically Similar Code

```bash
codedupes search ./src "parse json payload" --top-k 10
```

For full command/option semantics, see `docs/cli.md`.

## Override Semantic Instruction Prefix

By default, C2LLM task-specific prefixes are applied automatically. Override them
for experiments or custom retrieval behavior:

```bash
codedupes check ./src --instruction-prefix "Represent this code for duplicate detection: "
codedupes search ./src "parse json payload" --instruction-prefix "Represent this query for code lookup: "
```

## Fresh Colab/GPU preflight

Use one command to print runtime versions and CUDA availability:

```bash
python - <<'PY'
import platform
import torch
import transformers
import sentence_transformers
try:
    import deepspeed
    deepspeed_version = deepspeed.__version__
except Exception:
    deepspeed_version = "missing"

print("python", platform.python_version())
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("sentence-transformers", sentence_transformers.__version__)
print("deepspeed", deepspeed_version)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
PY
```

Pinned default semantic model revision:

```bash
codedupes info
```

Semantic runtime defaults:

- Uses `bfloat16` on CUDA (when supported) and on CPU (no `fp16` fallback).
- Default semantic `--batch-size` is `8`.
- On CUDA OOM, semantic embedding retries with progressively smaller GPU batches before CPU fallback.

## Threshold Tuning

Use a single threshold for both traditional and semantic:

```bash
codedupes check ./src --threshold 0.82
```

Set separate thresholds:

```bash
codedupes check ./src --semantic-threshold 0.84 --traditional-threshold 0.75
```

## Scope Control

Exclude private names:

```bash
codedupes check ./src --no-private
```

Exclude files with glob patterns:

```bash
codedupes check ./src --exclude "**/generated/**" --exclude "**/migrations/**"
```

By default, common artifact/cache directories are skipped automatically (for example `target`,
`node_modules`, `__pycache__`, `.venv`, `build`, `dist`, and similar tool outputs).

Include type stubs:

```bash
codedupes check ./src --include-stubs
```

## Unused Detection Modes

Default behavior is conservative and skips public top-level functions.

```bash
codedupes check ./src
```

Strict mode includes public functions:

```bash
codedupes check ./src --strict-unused
```

Disable unused detection:

```bash
codedupes check ./src --no-unused
```

## Reduce Semantic Noise In Test Suites

When auditing `tests/` directories, suppress semantic matches involving
pytest-style `test_*` functions:

```bash
codedupes check tests --suppress-test-semantic
```

## Full Run Verification Sequence

1. Baseline traditional-only:

```bash
codedupes check src --traditional-only
```

2. Semantic-only quick pass:

```bash
codedupes check src --semantic-only --min-lines 1 --batch-size 4
```

3. Full mixed run:

```bash
codedupes check src
```

Record model/revision, package versions, device, elapsed time, exit code, and whether semantic fallback warnings were emitted.

## Hybrid gate tuning workflow

Use the dedicated sweep harness + tracked synthetic corpus described in `docs/hybrid-tuning.md`.
Treat that corpus as a guardrail, then re-validate on at least one real repository before changing defaults.
