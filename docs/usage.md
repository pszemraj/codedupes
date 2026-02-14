# Usage Guide

## Install

```bash
pip install codedupes
```

Optional GPU extras:

```bash
pip install codedupes[gpu]
```

## Core Workflow

1. Run analysis on a package or file.
2. Review duplicate candidates.
3. Review potentially unused symbols.
4. Re-run with stricter/looser thresholds to tune sensitivity.

## Analyze A Project

```bash
codedupes check ./src
```

Analyze one file:

```bash
codedupes check ./src/module.py
```

## Use JSON For CI/Automation

```bash
codedupes check ./src --json
```

`check` returns:

- `0` when no duplicates/unused candidates are found
- `1` when findings exist or when an execution error occurs
- `2` for CLI usage/validation errors (invalid options/values)

## Control Rich Output Width

Use a wider output width for less wrapping in terminal tables:

```bash
codedupes check ./src --output-width 200
```

## Search Semantically Similar Code

```bash
codedupes search ./src "parse json payload" --top-k 10
```

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
