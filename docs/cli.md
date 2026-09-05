# CLI reference

The supported command-line entry point is `codedupes`. Examples assume it is installed and available on `PATH`.

See [Output and exit codes](output.md) for JSON and process status, [Polyglot language support](polyglot-languages.md) for extraction semantics, [Analysis defaults](analysis-defaults.md) for heuristics, [Model profiles](model-profiles.md) for semantic defaults, [Accelerators](accelerators.md) for device behavior, and [Embedding cache](caching.md) for persistent cache behavior.

## `codedupes check <path>`

Run duplicate and unused-code analysis.

Review the reported candidates, then adjust thresholds or scope if needed. See [Output and exit codes](output.md) for report modes and [hybrid gate tuning](hybrid-tuning.md) for calibration experiments.

Examples:

```bash
codedupes check ./src
codedupes check ./src --json --threshold 0.82
codedupes check ./src --semantic-only
codedupes check ./src --traditional-only --no-unused
codedupes check ./src --show-all
codedupes check ./src --fail-on all
codedupes check ./src/module.py
codedupes check ./src --semantic-threshold 0.84 --traditional-threshold 0.75
codedupes check ./src --exclude "**/generated/**" --exclude "**/migrations/**"
```

Options, in addition to the [shared options](#options-shared-by-check-and-search):

- `-t, --threshold <float>`: Shared threshold override for semantic and traditional checks (in single-method modes, it applies to the active method only)
- `--traditional-threshold <float>`: Override traditional (Jaccard) threshold only
- `--cross-language`: Also report semantic duplicate pairs across languages; see [comparison boundaries](polyglot-languages.md#fingerprints-and-comparison-boundaries)
- `--semantic-task <name>`: Semantic task mode for duplicate detection embeddings (default `semantic-similarity`)
- `--semantic-only`: Run semantic analysis only
- `--traditional-only`: Run traditional analysis only
- `--allow-semantic-fallback`: In default combined mode only, continue with full-scope traditional results if semantic backend loading/inference fails
- `--no-unused`: Disable unused-code detection
- `--strict-unused`: Include public non-method functions (module-level and nested) in unused checks
- `--suppress-test-semantic`: Suppress semantic duplicate matches involving `test_*` functions
- `--no-tiny-filter`: Disable tiny code-unit filtering for traditional duplicates
- `--tiny-cutoff <int>`: Tiny code-unit statement cutoff (exclusive) for traditional filtering (default `3`)
- `--show-all`: Also print raw traditional + raw semantic duplicate lists in combined mode
- `--full-table`: Disable table row truncation and print all rows in terminal output
- `--show-source`: Show truncated duplicate snippets
- `--fail-on <actionable|all|none>`: Select which reported findings produce exit code `1` (default `actionable`; see [exit codes](output.md#exit-codes))

## `codedupes search <path> "<query>"`

Run semantic search over extracted code units.

Examples:

```bash
codedupes search ./src "sum values in a list" --top-k 5
codedupes search ./src "normalize request payload" --json
codedupes search ./src "parse json payload" --semantic-threshold 0.6 --top-k 20
codedupes search ./src "refund validation" --search-document contextual --semantic-threshold 0.55
```

Options, in addition to the [shared options](#options-shared-by-check-and-search):

- `--top-k <int>`: Number of results (default `10`)
- `--threshold <float>`: Shared semantic threshold override
- `--semantic-task <name>`: Semantic task mode for query/document embeddings (default `code-retrieval`)
- `--search-document <source|contextual>`: Choose the [search document representation](python-api.md#semantic-query-search), source only by default. Contextual search requires an explicit `--semantic-threshold` or `--threshold`; tune the value against representative queries

## Options shared by `check` and `search`

rich-click groups command help into the following panels. Use `codedupes <command> -h` or `--help` for the rendered reference.

### Scope

```bash
codedupes check . --language python --language rust
codedupes search . "validate session token" --language js --language ts
```

- `--language <name>`: Restrict extraction to a language; repeat for multiple languages, or omit to auto-detect. See [supported files](polyglot-languages.md#supported-files) for names, aliases, and C header selection.
- `--no-private`: Exclude private units according to [language visibility rules](polyglot-languages.md#visibility-filtering)
- `--exclude <glob>`: Replace the default file globs (repeat for multiple patterns); see [extraction scope](analysis-defaults.md#extraction-scope-defaults)
- `--include-stubs`: Include `.pyi` files when scanning a directory (single-file `.pyi` targets are analyzed as given)

### Semantic model

```bash
codedupes check ./src --model embeddinggemma-300m
codedupes check ./src --instruction-prefix "Represent this code for duplicate detection: " --semantic-threshold 0.85
```

See [model profiles](model-profiles.md#semantic-task-defaults-and-choices) for task choices and when a custom configuration requires an explicit threshold.

- `--semantic-threshold <float>`: Flat semantic gate for every language; without it, `check` uses the model profile's calibrated [per-language gates](analysis-defaults.md#semantic-duplicate-gate-defaults) and `search` uses the profile search default
- `--semantic-unit-type <name>`: Semantic candidate unit type (`function`, `method`, `class`); repeat option to include multiple types (default `function, method`). Traditional duplicate matching always retains full extraction scope.
- `--min-statements <int>`: Minimum statement count for semantic candidate code units (default `3`). This does not narrow traditional duplicate matching.
- `--model <name>`: Embedding model alias, Hugging Face ID, or explicit path (absolute, `./`/`../`, or `~`) to a complete local `save_pretrained`/`hf download` directory (default `gte-modernbert-base`)
- `--model-revision <rev>`: Model revision/commit hash (defaults to the profile's pinned calibration commit for built-in models, unpinned otherwise)
- `--trust-remote-code` / `--no-trust-remote-code`: Allow or disallow model remote code execution
- `--instruction-prefix <text>`: Replace the model prompt for code/query embeddings (encode route is preserved)
- `--strict-revision-cache`: Key an unpinned hub model's cache revision to a resolved commit hash instead of the requested revision label, disabling caching when a branch/tag cannot be mapped offline

### Device

- `--device <name>`: Semantic inference device: `auto`, `cpu`, `cuda`, or `mps` (default `auto`; see [device selection](accelerators.md#device-selection))
- `--mps-fallback` / `--no-mps-fallback`: Enable or disable PyTorch CPU fallback for unsupported MPS operators
- `--mps-memory-fraction <float>`: Optional PyTorch MPS allocator fraction; see [memory policy](accelerators.md#mps-memory-policy-and-oom-recovery)
- `--batch-size <int>`: Embedding batch size (default `8`)

### Cache

- `--no-cache`: Disable the persistent on-disk embedding cache for this run

### Output

- `--output-width <int>`: Rich render width for non-JSON output (default `160`, min `80`)
- `--json`: Emit JSON instead of rich tables
- `-v, --verbose`: Verbose logs

## Environment variables

Every displayed option can be supplied with the `CODEDUPES_` prefix and its upper-case parameter name. Shared examples are `CODEDUPES_DEVICE=cpu`, `CODEDUPES_MODEL=...`, and `CODEDUPES_NO_CACHE=1`; command-line values take precedence. Command-specific examples include `CODEDUPES_FAIL_ON=all` for `check` and `CODEDUPES_SEARCH_DOCUMENT=contextual` for `search`. Cache-library controls such as `CODEDUPES_CACHE_DIR` and `CODEDUPES_CACHE_MAX_MB` are documented separately in [Embedding cache](caching.md#controls).

## `codedupes info`

Print installed runtime and parser versions, model aliases and effective defaults, analysis settings, device capabilities, and the embedding-cache summary. See [parser readiness](polyglot-languages.md#parser-readiness) and [accelerator precision](accelerators.md#precision-and-metal-environment-variables) for interpreting those fields.

## `codedupes cache info`

Print the embedding-cache summary plus per-model entry counts and a per-repo breakdown including orphan rows and the last complete manifest generation.

## `codedupes cache clear [--model <name>]`

Clear all cached embeddings or only entries for one model. See [Embedding cache](caching.md).

## Validation and mode notes

- Threshold values must be in `[0.0, 1.0]`
- `--batch-size` and `--top-k` must be greater than `0`
- `--min-statements` and `--tiny-cutoff` must be greater than or equal to `0`
- `--show-all` and `--allow-semantic-fallback` are only valid in default combined `check` mode (not with `--semantic-only` or `--traditional-only`)
- `--json` rejects rich-only display controls: `--show-source`, `--full-table`, `--verbose`, and explicit `--output-width`
- `--semantic-only` and `--traditional-only` are mutually exclusive
- `--no-unused` and `--strict-unused` are mutually exclusive
- `--trust-remote-code` and `--no-trust-remote-code` are mutually exclusive
- `--mps-fallback` and `--no-mps-fallback` are mutually exclusive
- Explicit semantic-analysis controls are rejected with `--traditional-only`, including model/task, candidate-scope, device/runtime options, and `--strict-revision-cache`. `--no-cache` is accepted as a harmless no-op.
- Explicit traditional-analysis controls are rejected with `--semantic-only`: `--traditional-threshold`, `--no-tiny-filter`, and `--tiny-cutoff`

To investigate a surprising combined result, compare `--traditional-only`, `--semantic-only`, and the default run. Add `--verbose` for model-loading, device-resolution, and fallback logs. See [semantic candidate rules](analysis-defaults.md#semantic-candidate-defaults) for context-window exclusions and [Output and exit codes](output.md) for diagnostics and failure behavior.
