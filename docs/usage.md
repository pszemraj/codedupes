# Usage Guide

Install `codedupes` first, then use these workflows to analyze, search, and tune a project. The
[CLI reference](cli.md) lists every option.

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

See [Output and exit codes](output.md) for JSON structure and process status.

## Control Rich Output Width

Use a wider output width for less wrapping in terminal tables:

```bash
codedupes check ./src --output-width 200
```

## Search Semantically Similar Code

```bash
codedupes search ./src "parse json payload" --top-k 10
```

## Select Model And Task

Choose model aliases or raw HuggingFace IDs:

```bash
codedupes check ./src --model gte-modernbert-base
codedupes check ./src --model embeddinggemma-300m
```

Set task behavior explicitly:

```bash
codedupes check ./src --semantic-task semantic-similarity
codedupes search ./src "parse json payload" --semantic-task code-retrieval
```

See [Model profiles](model-profiles.md) for aliases, thresholds, and task behavior.

## Apple Silicon / MPS

```bash
codedupes check ./src --device mps
codedupes search ./src "parse json payload" --device mps
```

On a memory-constrained machine, begin with a conservative allocator cap and a smaller batch:

```bash
codedupes check ./src --device mps --mps-memory-fraction 0.9 --batch-size 4
```

See [Accelerators](accelerators.md) for device resolution, memory limits, and OOM recovery.

## Override Semantic Instruction Prefix

By default, model-profile task prompts are applied automatically when needed. Override
with a fixed prefix for experiments or custom retrieval behavior:

```bash
codedupes check ./src --instruction-prefix "Represent this code for duplicate detection: "
codedupes search ./src "parse json payload" --instruction-prefix "Represent this query for code lookup: "
```

## Threshold Tuning

Use a single threshold override for both traditional and semantic:

```bash
codedupes check ./src --threshold 0.82
```

Set separate thresholds:

```bash
codedupes check ./src --semantic-threshold 0.84 --traditional-threshold 0.75
```

Raise the search threshold to keep only stronger matches:

```bash
codedupes search ./src "parse json payload" --semantic-threshold 0.6 --top-k 20
```

See [Model profiles](model-profiles.md) for model-specific check and search thresholds.

## Scope Control

Exclude private names:

```bash
codedupes check ./src --no-private
```

Exclude files with glob patterns:

```bash
codedupes check ./src --exclude "**/generated/**" --exclude "**/migrations/**"
```

Include type stubs:

```bash
codedupes check ./src --include-stubs
```

Control semantic candidate unit types:

```bash
codedupes check ./src
codedupes check ./src --semantic-unit-type function --semantic-unit-type method --semantic-unit-type class
```

See [Analysis defaults](analysis-defaults.md) for extraction and candidate scope.

## Reduce Boilerplate Duplicate Noise

```bash
codedupes check ./src --no-tiny-filter
codedupes check ./src --tiny-cutoff 4 --tiny-near-jaccard-min 0.95
```

See [Analysis defaults](analysis-defaults.md) for tiny-pair filtering behavior.

## Unused Detection Modes

```bash
codedupes check ./src
codedupes check ./src --strict-unused
codedupes check ./src --no-unused
```

See [Analysis defaults](analysis-defaults.md#potentially-unused-defaults) for the
reference graph, suppressions, strict-mode behavior, and limitations.

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

Record model/revision, package versions, requested and resolved device, elapsed time, exit code, and whether unsupported-op or OOM fallback warnings were emitted.

## Hybrid gate tuning workflow

Use the sweep harness and tracked corpus described in [Hybrid gate tuning](hybrid-tuning.md).
