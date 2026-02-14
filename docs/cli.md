# CLI Reference

This document is the source of truth for CLI commands, flags, and option defaults.
For JSON schemas and exit-code semantics, see
[docs/output.md](https://github.com/pszemraj/codedupes/blob/main/docs/output.md) (the source of truth for output behavior).
For analysis-behavior defaults (semantic candidate scope, tiny-traditional filtering, hybrid gates), see
[docs/analysis-defaults.md](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md).
For semantic model aliases/profile defaults/task behavior, see
[docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md).

## Commands

## `codedupes check <path>`

Run duplicate and unused-code analysis.

Default behavior is hybrid-first:

- one synthesized `Hybrid Duplicates` list (combined traditional + semantic evidence)
- likely dead-code candidates

Use `--show-all` to additionally print raw traditional and raw semantic duplicate lists.

Examples:

```bash
codedupes check ./src
codedupes check ./src --json --threshold 0.82
codedupes check ./src --semantic-only
codedupes check ./src --traditional-only --no-unused
codedupes check ./src --show-all
```

Options:

- `-t, --threshold <float>`: Shared threshold override for semantic and traditional checks (in single-method modes, it applies to the active method only)
- `--semantic-threshold <float>`: Override semantic threshold only
- `--traditional-threshold <float>`: Override traditional (Jaccard) threshold only
- `--semantic-task <name>`: Semantic task mode for duplicate detection embeddings (default `semantic-similarity`)
- `--semantic-only`: Run semantic analysis only
- `--traditional-only`: Run traditional analysis only
- `--allow-semantic-fallback`: In default combined mode only, continue with scoped traditional results if semantic backend loading/inference fails
- `--no-unused`: Disable unused-code detection
- `--strict-unused`: Include public top-level functions in unused checks
- `--suppress-test-semantic`: Suppress semantic duplicate matches involving `test_*` functions
- `--semantic-unit-type <name>`: Semantic candidate unit type (`function`, `method`, `class`); repeat option to include multiple types (default `function, method`). In default combined mode this also narrows traditional duplicate scope.
- `--no-tiny-filter`: Disable tiny function/method filtering for traditional duplicates
- `--tiny-cutoff <int>`: Tiny function/method statement cutoff (exclusive) for traditional filtering (default `3`)
- `--tiny-near-jaccard-min <float>`: Minimum Jaccard similarity to keep tiny near-duplicate pairs (default `0.93`)
- `--show-all`: Also print raw traditional + raw semantic duplicate lists in combined mode
- `--full-table`: Disable table row truncation and print all rows in terminal output
- `--min-lines <int>`: Minimum statement count for semantic candidate code units (default `3`). In default combined mode this also narrows traditional duplicate scope.
- `--model <name>`: Embedding model alias or HuggingFace ID (default `gte-modernbert-base`)
- `--instruction-prefix <text>`: Override default semantic instruction prefix for code/query embeddings
- `--model-revision <rev>`: Model revision/commit hash (default `auto`; profile-specific behavior)
- `--trust-remote-code`: Allow model remote code execution
- `--no-trust-remote-code`: Disallow model remote code execution
- `--batch-size <int>`: Embedding batch size (default `8`)
- `--no-private`: Exclude private (`_name`) functions/classes
- `--exclude <glob>`: Add file path glob pattern(s) to exclude (repeat option for multiple patterns). Built-in test/artifact excludes still apply.
- `--include-stubs`: Include `*.pyi` files
- `--output-width <int>`: Rich render width for non-JSON output (default `160`, min `80`)
- `--show-source`: Show truncated duplicate snippets
- `--json`: Emit JSON instead of rich tables
- `-v, --verbose`: Verbose logs

## `codedupes search <path> "<query>"`

Run semantic search over extracted code units.

Examples:

```bash
codedupes search ./src "sum values in a list" --top-k 5
codedupes search ./src "normalize request payload" --json
```

Options:

- `--top-k <int>`: Number of results (default `10`)
- `--model <name>`: Embedding model alias or HuggingFace ID (default `gte-modernbert-base`)
- `--semantic-task <name>`: Semantic task mode for query/document embeddings (default `code-retrieval`)
- `--semantic-unit-type <name>`: Semantic candidate unit type (`function`, `method`, `class`); repeat option to include multiple types (default `function, method`)
- `--instruction-prefix <text>`: Override default semantic instruction prefix for code/query embeddings
- `--model-revision <rev>`: Model revision/commit hash (default `auto`; profile-specific behavior)
- `--trust-remote-code`: Allow model remote code execution
- `--no-trust-remote-code`: Disallow model remote code execution
- `--threshold <float>`: Shared semantic threshold override
- `--semantic-threshold <float>`: Override semantic threshold
- `--batch-size <int>`: Embedding batch size (default `8`)
- `--min-lines <int>`: Minimum statement count for semantic candidate code units (default `3`)
- `--no-private`: Exclude private (`_name`) functions/classes
- `--exclude <glob>`: Add file path glob pattern(s) to exclude (repeat option for multiple patterns). Built-in test/artifact excludes still apply.
- `--include-stubs`: Include `*.pyi` files
- `--output-width <int>`: Rich render width for non-JSON output (default `160`, min `80`)
- `--json`: Emit JSON instead of rich tables
- `-v, --verbose`: Verbose logs

## `codedupes info`

Print version and default settings.

## Validation and mode notes

- Threshold values must be in `[0.0, 1.0]`
- `--batch-size` and `--top-k` must be greater than `0`
- `--min-lines` must be greater than or equal to `0`
- `--output-width` must be at least `80`
- `--show-all` is only valid in default combined `check` mode (not with `--semantic-only` or `--traditional-only`)
- Default combined `check` fails if semantic backend fails; opt in to degraded combined fallback with `--allow-semantic-fallback`
- In `--json` mode, output is machine-parseable JSON only; warning text is surfaced via
  `summary.semantic_fallback` and `summary.semantic_fallback_reason` when fallback happens.
- `--json` rejects rich-only display controls: `--show-source`, `--full-table`, `--verbose`, and explicit `--output-width`
- `--semantic-only` and `--traditional-only` bypass hybrid synthesis and show raw method outputs
- `--semantic-only` and `--traditional-only` are mutually exclusive
- `--trust-remote-code` and `--no-trust-remote-code` are mutually exclusive
- `search` applies semantic threshold filtering before returning `top-k` matches
- Contradictory mode-specific options are rejected at parse time for the selected workflow

Built-in model aliases and model-profile defaults are documented in
[docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md).
You can also inspect effective model defaults in your environment via:

```bash
codedupes info
```

For JSON payloads and complete exit-code semantics, see
[docs/output.md](https://github.com/pszemraj/codedupes/blob/main/docs/output.md).
