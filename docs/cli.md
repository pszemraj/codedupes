# CLI Reference

## Commands

## `codedupes check <path>`

Run duplicate and unused-code analysis.

Examples:

```bash
codedupes check ./src
codedupes check ./src --json --threshold 0.82
codedupes check ./src --semantic-only
codedupes check ./src --traditional-only --no-unused
```

Options:

- `-t, --threshold <float>`
- `--semantic-threshold <float>`
- `--traditional-threshold <float>`
- `--semantic-only`
- `--traditional-only`
- `--no-unused`
- `--strict-unused`
- `--min-lines <int>`
- `--model <name>`
- `--batch-size <int>`
- `--no-private`
- `--exclude <glob> [<glob> ...]`
- `--include-stubs`
- `--show-source`
- `--json`
- `-v, --verbose`

## `codedupes search <path> "<query>"`

Run semantic search over extracted code units.

Examples:

```bash
codedupes search ./src "sum values in a list" --top-k 5
codedupes search ./src "normalize request payload" --json
```

Options:

- `--top-k <int>`
- `--model <name>`
- `--threshold <float>`
- `--semantic-threshold <float>`
- `--batch-size <int>`
- `--min-lines <int>`
- `--no-private`
- `--exclude <glob> [<glob> ...]`
- `--include-stubs`
- `--json`
- `-v, --verbose`

## `codedupes info`

Print version and default settings.

## Global Behavior Notes

- Threshold values must be in `[0.0, 1.0]`.
- `--batch-size` and `--top-k` must be greater than `0`.
- `--min-lines` must be greater than or equal to `0`.
