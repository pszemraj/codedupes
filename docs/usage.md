# Usage Guide

This guide focuses on practical workflows.
Install and dependency setup are defined in
[docs/install.md](https://github.com/pszemraj/codedupes/blob/main/docs/install.md).
Flag defaults and validation rules are defined in
[docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md).
Analysis behavior defaults are defined in
[docs/analysis-defaults.md](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md).
Semantic model aliases/profile defaults/task behavior are defined in
[docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md).
JSON schema and exit codes are defined in
[docs/output.md](https://github.com/pszemraj/codedupes/blob/main/docs/output.md).

## Install

See [docs/install.md](https://github.com/pszemraj/codedupes/blob/main/docs/install.md).

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

Print full terminal tables without the default 20-row truncation:

```bash
codedupes check ./src --full-table
```

Analyze one file:

```bash
codedupes check ./src/module.py
```

## Use JSON For CI/Automation

```bash
codedupes check ./src --json
```

See [docs/output.md](https://github.com/pszemraj/codedupes/blob/main/docs/output.md) for authoritative JSON
structure and exit-code semantics.

## Control Rich Output Width

Use a wider output width for less wrapping in terminal tables:

```bash
codedupes check ./src --output-width 200
```

## Search Semantically Similar Code

```bash
codedupes search ./src "parse json payload" --top-k 10
```

For full command/option semantics, see
[docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md).

## Select Model And Task

Choose model aliases or raw HuggingFace IDs:

```bash
codedupes check ./src --model gte-modernbert-base
codedupes check ./src --model c2llm-0.5b
codedupes check ./src --model embeddinggemma-300m
```

Set task behavior explicitly:

```bash
codedupes check ./src --semantic-task semantic-similarity
codedupes search ./src "parse json payload" --semantic-task code-retrieval
```

## Override Semantic Instruction Prefix

By default, model-profile task prompts are applied automatically when needed. Override
with a fixed prefix for experiments or custom retrieval behavior:

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

Model profiles and revision defaults:

```bash
codedupes info
```

Semantic runtime defaults are documented in
[docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md).

## Threshold Tuning

Use a single threshold override for both traditional and semantic:

```bash
codedupes check ./src --threshold 0.82
```

Set separate thresholds:

```bash
codedupes check ./src --semantic-threshold 0.84 --traditional-threshold 0.75
```

Search applies threshold filtering before top-k:

```bash
codedupes search ./src "parse json payload" --semantic-threshold 0.9 --top-k 20
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

Control semantic candidate unit types:

```bash
codedupes check ./src
codedupes check ./src --semantic-unit-type function --semantic-unit-type method --semantic-unit-type class
```

Default semantic candidate behavior is documented in
[docs/analysis-defaults.md](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md).
Use `--semantic-unit-type class` when you explicitly want class-level semantic embeddings.

## Reduce Boilerplate Duplicate Noise

Traditional duplicate detection filters tiny function/method wrappers by default.
The authoritative default semantics are in
[docs/analysis-defaults.md](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md).

Override behavior when needed:

```bash
codedupes check ./src --no-tiny-filter
codedupes check ./src --tiny-cutoff 4 --tiny-near-jaccard-min 0.95
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

Use the dedicated sweep harness + tracked synthetic corpus described in
[docs/hybrid-tuning.md](https://github.com/pszemraj/codedupes/blob/main/docs/hybrid-tuning.md).
Treat that corpus as a guardrail, then re-validate on at least one real repository before changing defaults.
