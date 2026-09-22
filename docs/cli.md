# CLI reference

The supported command-line entry point is `codedupes`. Install it with the [installation guide](install.md), then run `codedupes info` to confirm parser and device readiness. Examples assume the command is available on `PATH`.

`codedupes --version` prints the installed version. Running `codedupes` without a subcommand prints help and exits `2`.

See linked topic documentation in the relevant option descriptions for behavior beyond command syntax.

## `codedupes check <path>`

Run duplicate and unused-code analysis. [Analysis defaults](analysis-defaults.md#what-a-default-check-does) describe the combined scan and candidate tiers.

Examples:

```bash
codedupes check ./src
codedupes check ./src --json --threshold 0.82
codedupes check ./src --semantic-only
# Fast structural/token scan without semantic model inference.
codedupes check ./src --traditional-only --no-unused
codedupes check ./src --include-review
codedupes check ./src --show-all
codedupes check ./src --json --max-duplicates 50
# Every default-visible finding, not just the top 20 duplicates and 20 unused units.
codedupes check ./src --json --max-duplicates all --max-unused all
codedupes check ./src --fail-on all
codedupes check ./src/module.py
codedupes check ./src --semantic-threshold 0.84 --traditional-threshold 0.75
codedupes check ./src --exclude "**/generated/**" --exclude "**/migrations/**"
codedupes check tests --no-default-excludes --no-unused
codedupes check ./src --json --show-source --source-lines all
codedupes check ./src --traditional-only --show-diff
```

Options, in addition to the [shared options](#options-shared-by-check-and-search):

- `-t, --threshold <float>`: Shared threshold override for semantic and traditional checks (in single-method modes, it applies to the active method only)
- `--traditional-threshold <float>`: Override the [traditional Jaccard threshold](analysis-defaults.md#traditional-duplicate-defaults) only
- `--cross-language`: Also report semantic duplicate pairs across languages; see [comparison boundaries](polyglot-languages.md#fingerprints-and-comparison-boundaries)
- `--semantic-task <name>`: Duplicate embedding task; see [task defaults and choices](model-profiles.md#semantic-task-defaults-and-choices)
- `--semantic-only`: Use only semantic matching for duplicate detection
- `--traditional-only`: Use only traditional matching for duplicate detection
- `--allow-semantic-fallback`: Enable [combined-mode fallback](output.md#exit-codes)
- `--no-unused`: Disable unused-code detection
- `--strict-unused`: Also report unreferenced public functions and public methods; see the [unused-code policy](analysis-defaults.md#potentially-unused-defaults)
- `--suppress-test-semantic`: Suppress semantic duplicate matches involving `test_*` functions
- `--no-tiny-filter`: Disable tiny code-unit filtering for traditional duplicates
- `--tiny-cutoff <int>`: Override the [traditional tiny-filter cutoff](analysis-defaults.md#traditional-duplicate-defaults)
- `--include-review`, `--show-all`: Expand the [reported findings](output.md#report-selection) and lift the default report caps
- `--max-duplicates <N|all>`: Cap the [reported duplicate findings](output.md#report-selection) at `N` (default `20`; exact families first and counted once, then actionable tiers) or remove the cap with `all`
- `--max-unused <N|all>`: Cap the [reported unused units](output.md#report-selection) at `N` (default `20`, largest line span first) or remove the cap with `all`
- `--full-table`: Print all rows in the raw `--show-all` duplicate tables and lift the default `--max-duplicates` and `--max-unused` caps
- `--show-source`: Show a bounded source snippet per reported unit, in terminal panels and (unlike other display controls) also as a `source` field on every JSON unit record
- `--source-lines <N|all>`: Cap each shown snippet at `N` lines (default `40`) or remove the cap with `all`; implies `--show-source`
- `--show-diff`: Show a unified diff per duplicate pair, bounded by `--source-lines`; `structural_hash` families diff each member against the first, `token_hash` families print nothing extra (token-identical already)
- `--fail-on <actionable|all|none>`: Select the [finding exit policy](output.md#exit-codes)
- `--fail-on-incomplete`: Also exit `1` when the [analysis did not complete](output.md#exit-codes), independent of `--fail-on` (including `none`)

Single-method flags leave unused-code detection enabled; add `--no-unused` to disable it.

### Single-file targets

`codedupes check <file>` only compares code units within that one file, so a duplicate of it living elsewhere in the project is not reported. Unused-code detection is not limited the same way: it resolves a [project-wide reference root](analysis-defaults.md#extraction-scope-defaults) for the target (the nearest `pyproject.toml`, else the git work tree, else the file's own directory) and parses every Python file under it, test files included, so a call from elsewhere in the project still counts.

```text
# Cross-file duplicate detection for one file against the rest of a project
# is not available in this release; scan the project root instead.
codedupes check <root> --focus <file>
```

## `codedupes search <path> "<query>"`

Run semantic search over extracted code units.

Search indexes the chosen path for this invocation, returning functions and methods by default. Use `--semantic-unit-type` to include classes; the persistent cache can reuse embeddings between invocations.

Examples:

```bash
codedupes search ./src "sum values in a list" --top-k 5
codedupes search ./src "normalize request payload" --json
codedupes search ./src "normalize request payload" --result-level file --top-k 5
codedupes search ./src "parse json payload" --semantic-threshold 0.6 --top-k 20
codedupes search ./src "refund validation" --search-document contextual --semantic-threshold 0.55
```

Options, in addition to the [shared options](#options-shared-by-check-and-search):

- `--result-level <unit|file>`: Return individual code units (default `unit`) or group matches into files
- `--top-k <int>`: Maximum results at the selected level: code units or distinct files (default `10`)
- `--threshold <float>`: Shared semantic threshold override
- `--semantic-task <name>`: Query/document embedding task; see [task defaults and choices](model-profiles.md#semantic-task-defaults-and-choices)
- `--search-document <source|contextual>`: Choose the [search document representation](python-api.md#semantic-query-search)

For file-level grouping and `--top-k` order, see [file search output](output.md#file-search).

## Options shared by `check` and `search`

Use `codedupes <command> -h` or `--help` for rendered option help.

### Scope

```bash
codedupes check . --language python --language rust
codedupes search . "validate session token" --language js --language ts
```

- `--language <name>`: Restrict extraction to a language; repeat for multiple languages, or omit to auto-detect. See [supported files](polyglot-languages.md#supported-files) for names, aliases, and C header selection.
- `--no-private`: Exclude private units according to [language visibility rules](polyglot-languages.md#visibility-filtering)
- `--exclude <name|glob>`: Add a quoted exclusion pattern; repeat for multiple patterns. See [pattern matching and scope](analysis-defaults.md#extraction-scope-defaults)
- `--no-default-excludes`: Disable [default test-file exclusions](analysis-defaults.md#extraction-scope-defaults)
- `--no-gitignore`: Scan paths git ignores; by default a directory scan inside a git work tree [skips them](analysis-defaults.md#extraction-scope-defaults)
- `--include-stubs`: Include `.pyi` files when scanning a directory (single-file `.pyi` targets are analyzed as given)

### Semantic model

```bash
codedupes check ./src --model embeddinggemma-300m
codedupes check ./src --instruction-prefix "Represent this code for duplicate detection: " --semantic-threshold 0.85
```

- `--semantic-threshold <float>`: Override the [semantic gate](analysis-defaults.md#semantic-duplicate-gate-defaults) or [search floor](model-profiles.md#built-in-profiles)
- `--threshold-profile <auto|generic|embeddinggemma-300m|gte-modernbert-base>`: Choose [threshold defaults](model-profiles.md#choosing-threshold-defaults)
- `--semantic-unit-type <name>`: Semantic candidate unit type (`function`, `method`, `class`); repeat for multiple types. See [candidate defaults](analysis-defaults.md#semantic-candidate-defaults)
- `--min-statements <int>`: Override the [semantic candidate statement minimum](analysis-defaults.md#semantic-candidate-defaults)
- `--model <name>`: Select a [model alias, Hub ID, or explicit local path](model-profiles.md#alias-resolution-rules)
- `--model-revision <rev>`: Override the [profile's model revision](model-profiles.md#built-in-profiles)
- `--trust-remote-code` / `--no-trust-remote-code`: Allow or disallow model remote code execution
- `--instruction-prefix <text>`: Replace the model prompt for code/query embeddings (encode route is preserved)
- `--strict-revision-cache` / `--loose-revision-cache`: Require concrete Hub revisions before cache reuse (default), or opt into potentially stale label-keyed warm hits

### Device

- `--device <name>`: Semantic inference device: `auto`, `cpu`, `cuda`, or `mps` (default `auto`; see [device selection](accelerators.md#device-selection))
- `--mps-fallback` / `--no-mps-fallback`: Enable or disable PyTorch CPU fallback for unsupported MPS operators
- `--mps-memory-fraction <float>`: Optional PyTorch MPS allocator fraction; see [memory policy](accelerators.md#accelerator-oom-recovery-and-mps-memory-policy)
- `--batch-size <int>`: Embedding batch size (default `8`)

### Cache

- `--no-cache`: Disable the persistent on-disk embedding cache for this run

### Output

- `--output-width <int>`: Maximum Rich render width for non-JSON output (default `160`, min `80`); capped at the terminal width, even on narrower terminals. Redirected output uses the requested width. Also accepted by `info`, `cache info`, and `cache clear`
- `--json`: Emit JSON instead of rich tables
- `-v, --verbose`: Verbose logs

## Environment variables

CLI options are configured through command-line flags; automatic `CODEDUPES_*` option overrides are disabled. Explicit library-level [cache controls](caching.md#controls) and [accelerator controls](accelerators.md#precision-and-metal-environment-variables) remain supported.

## `codedupes info`

Show the tool, runtime, device, model, and language summary. Add `-v` or `--verbose` for diagnostics, parser status, defaults, profiles, and cache details; device errors remain visible in the compact overview. See [parser readiness](polyglot-languages.md#parser-readiness) and [accelerator precision](accelerators.md#precision-and-metal-environment-variables) for interpretation.

## `codedupes cache info`

Display the embedding-cache summary plus per-model entry counts and a per-repo breakdown in Rich panels, including orphan rows and the last complete manifest generation.

## `codedupes cache clear [--model <name>]`

Clear all cached embeddings or only entries for one model. An empty or whitespace-only `--model` is a usage error (exit `2`) and deletes nothing; omit the option to clear all models. A failure to construct the cache, an outright clear failure, or a best-effort clear that leaves any deletion failed all exit `3` (`cache info` uses the same code for its own construction failure); see [exit codes](output.md#exit-codes). See [Embedding cache](caching.md).

## Validation and mode notes

- `check` threshold values must be in `[0.0, 1.0]`; `search --threshold`
  accepts any finite value, including a negative similarity floor
- `--semantic-threshold` and `--traditional-threshold` override `--threshold` for their respective methods
- `--batch-size` and `--top-k` must be greater than `0`; `--max-duplicates` and `--max-unused` take a positive integer or `all`
- `--min-statements` and `--tiny-cutoff` must be greater than or equal to `0`
- `--include-review`, `--show-all`, and `--allow-semantic-fallback` are only valid in default combined `check` mode (not with `--semantic-only` or `--traditional-only`)
- `--json` rejects rich-only display controls: `--show-diff`, `--full-table`, `--verbose`, and explicit `--output-width`; `--show-source`/`--source-lines` are accepted and add `source` to JSON unit records instead
- `--semantic-only` and `--traditional-only` are mutually exclusive
- `--no-unused` and `--strict-unused` are mutually exclusive
- `--trust-remote-code` and `--no-trust-remote-code` are mutually exclusive
- `--mps-fallback` and `--no-mps-fallback` are mutually exclusive
- Explicit semantic-analysis controls are rejected with `--traditional-only`, including model/task, candidate-scope, device/runtime options, and either revision-cache policy flag. `--no-cache` is accepted as a harmless no-op.
- Explicit traditional-analysis controls are rejected with `--semantic-only`: `--traditional-threshold`, `--no-tiny-filter`, and `--tiny-cutoff`

To investigate a surprising combined result, compare `--traditional-only`, `--semantic-only`, and the default run. Add `--verbose` for model-loading, device-resolution, and fallback logs. See [semantic candidate rules](analysis-defaults.md#semantic-candidate-defaults) for candidate selection and long-input handling, and [Output and exit codes](output.md) for diagnostics and failure behavior.
