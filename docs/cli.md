# CLI Reference

This document is the source of truth for CLI commands, flags, and option defaults.
For JSON schemas and exit-code semantics, see
[docs/output.md](https://github.com/pszemraj/codedupes/blob/main/docs/output.md) (the source of truth for output behavior).

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

- `-t, --threshold <float>`: Shared threshold override for semantic and traditional checks
- `--semantic-threshold <float>`: Override semantic threshold only
- `--traditional-threshold <float>`: Override traditional (Jaccard) threshold only
- `--semantic-task <name>`: Semantic task mode for duplicate detection embeddings (default `semantic-similarity`)
- `--semantic-only`: Run semantic analysis only
- `--traditional-only`: Run traditional analysis only
- `--no-unused`: Disable unused-code detection
- `--strict-unused`: Include public top-level functions in unused checks
- `--suppress-test-semantic`: Suppress semantic duplicate matches involving `test_*` functions
- `--semantic-unit-type <name>`: Semantic candidate unit type (`function`, `method`, `class`); repeat option to include multiple types (default `function, method`)
- `--no-tiny-filter`: Disable tiny function/method filtering for traditional duplicates
- `--tiny-cutoff <int>`: Tiny function/method statement cutoff (exclusive) for traditional filtering (default `3`)
- `--tiny-near-jaccard-min <float>`: Minimum Jaccard similarity to keep tiny near-duplicate pairs (default `0.93`)
- `--show-all`: Also print raw traditional + raw semantic duplicate lists in combined mode
- `--full-table`: Disable table row truncation and print all rows in terminal output
- `--min-lines <int>`: Minimum statement count for semantic candidate code units (default `3`)
- `--model <name>`: Embedding model alias or HuggingFace ID (default `gte-modernbert-base`)
- `--instruction-prefix <text>`: Override default semantic instruction prefix for code/query embeddings
- `--model-revision <rev>`: Model revision/commit hash (default `auto`; profile-specific behavior)
- `--trust-remote-code / --no-trust-remote-code`: Allow/disallow model remote code execution
- `--batch-size <int>`: Embedding batch size (default `8`)
- `--no-private`: Exclude private (`_name`) functions/classes
- `--exclude <glob>`: Exclude file path glob pattern (repeat option for multiple patterns)
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
- `--trust-remote-code / --no-trust-remote-code`: Allow/disallow model remote code execution
- `--threshold <float>`: Shared semantic threshold override
- `--semantic-threshold <float>`: Override semantic threshold
- `--batch-size <int>`: Embedding batch size (default `8`)
- `--min-lines <int>`: Minimum statement count for semantic candidate code units (default `3`)
- `--no-private`: Exclude private (`_name`) functions/classes
- `--exclude <glob>`: Exclude file path glob pattern (repeat option for multiple patterns)
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
- Default `check` behavior degrades to non-semantic analysis if semantic backend fails
- `--semantic-only` and `--traditional-only` bypass hybrid synthesis and show raw method outputs
- `search` applies semantic threshold filtering before returning `top-k` matches

Built-in model aliases:

- `gte-modernbert-base` -> `Alibaba-NLP/gte-modernbert-base`
- `c2llm-0.5b` -> `codefuse-ai/C2LLM-0.5B`
- `embeddinggemma-300m` -> `unsloth/embeddinggemma-300m`

For JSON payloads and complete exit-code semantics, see
[docs/output.md](https://github.com/pszemraj/codedupes/blob/main/docs/output.md).
