# CLI Reference

See [Output and exit codes](output.md) for JSON and process status, [Polyglot language support](polyglot-languages.md) for extraction semantics, [Analysis defaults](analysis-defaults.md) for heuristics, [Model profiles](model-profiles.md) for semantic defaults, [Accelerators](accelerators.md) for device behavior, and [Embedding cache](caching.md) for persistent cache behavior.

## `codedupes check <path>`

Run duplicate and unused-code analysis.

The default combined mode reports synthesized hybrid duplicates and unused candidates. See [Output and exit codes](output.md) for report modes.

Examples:

```bash
codedupes check ./src
codedupes check ./src --json --threshold 0.82
codedupes check ./src --semantic-only
codedupes check ./src --traditional-only --no-unused
codedupes check ./src --show-all
codedupes check ./src --device mps --mps-memory-fraction 0.9
```

Options, in addition to the [shared options](#options-shared-by-check-and-search):

- `-t, --threshold <float>`: Shared threshold override for semantic and traditional checks (in single-method modes, it applies to the active method only)
- `--traditional-threshold <float>`: Override traditional (Jaccard) threshold only
- `--cross-language`: Also report semantic duplicate pairs across languages (uncalibrated; a mixed pair is held to the looser of its two language gates)
- `--semantic-task <name>`: Semantic task mode for duplicate detection embeddings (default `semantic-similarity`)
- `--semantic-only`: Run semantic analysis only
- `--traditional-only`: Run traditional analysis only
- `--allow-semantic-fallback`: In default combined mode only, continue with scoped traditional results if semantic backend loading/inference fails
- `--no-unused`: Disable unused-code detection
- `--strict-unused`: Include public non-method functions (module-level and nested) in unused checks
- `--suppress-test-semantic`: Suppress semantic duplicate matches involving `test_*` functions
- `--no-tiny-filter`: Disable tiny function/method filtering for traditional duplicates
- `--tiny-cutoff <int>`: Tiny function/method statement cutoff (exclusive) for traditional filtering (default `3`)
- `--tiny-near-jaccard-min <float>`: Minimum Jaccard similarity to keep tiny near-duplicate pairs (default `0.93`)
- `--show-all`: Also print raw traditional + raw semantic duplicate lists in combined mode
- `--full-table`: Disable table row truncation and print all rows in terminal output
- `--show-source`: Show truncated duplicate snippets

## `codedupes search <path> "<query>"`

Run semantic search over extracted code units.

Examples:

```bash
codedupes search ./src "sum values in a list" --top-k 5
codedupes search ./src "normalize request payload" --json
codedupes search ./src "normalize request payload" --device mps
```

Options, in addition to the [shared options](#options-shared-by-check-and-search):

- `--top-k <int>`: Number of results (default `10`)
- `--threshold <float>`: Shared semantic threshold override
- `--semantic-task <name>`: Semantic task mode for query/document embeddings (default `code-retrieval`)

## Options shared by `check` and `search`

- `--language <name>`: Restrict extraction to a language; repeat for multiple languages. Canonical values are `python`, `c`, `rust`, `javascript`, and `typescript`; aliases `py`, `rs`, `js`, `jsx`, `ts`, and `tsx` are accepted. Omit the option to auto-detect all supported languages. Explicit `--language c` also opts ambiguous `.h` files into C parsing.
- `--semantic-threshold <float>`: Flat semantic gate for every language; without it, `check` uses the model profile's calibrated [per-language gates](analysis-defaults.md#semantic-duplicate-gate-defaults) and `search` uses the profile search default
- `--semantic-unit-type <name>`: Semantic candidate unit type (`function`, `method`, `class`); repeat option to include multiple types (default `function, method`). In default combined `check` mode this also narrows traditional duplicate scope.
- `--min-statements <int>`: Minimum statement count for semantic candidate code units (default `3`). In default combined `check` mode this also narrows traditional duplicate scope.
- `--model <name>`: Embedding model alias, Hugging Face ID, or explicit path (absolute, `./`/`../`, or `~`) to a complete local `save_pretrained`/`hf download` directory (default `gte-modernbert-base`)
- `--model-revision <rev>`: Model revision/commit hash (defaults to the profile's pinned calibration commit for built-in models, unpinned otherwise)
- `--trust-remote-code` / `--no-trust-remote-code`: Allow or disallow model remote code execution
- `--device <name>`: Semantic inference device: `auto`, `cpu`, `cuda`, or `mps` (default `auto`; priority CUDA, then MPS, then CPU)
- `--mps-fallback` / `--no-mps-fallback`: Enable or disable PyTorch CPU fallback for unsupported MPS operators
- `--mps-memory-fraction <float>`: Optional PyTorch MPS allocator fraction in `(0, 2]`; `0` is rejected as unsafe
- `--instruction-prefix <text>`: Replace the model prompt for code/query embeddings (encode route is preserved)
- `--batch-size <int>`: Embedding batch size (default `8`)
- `--no-private`: Exclude private (`_name`) functions/classes
- `--exclude <glob>`: Replace the default test-file globs with one or more file path globs (repeat for multiple patterns). Built-in artifact-directory exclusions still apply.
- `--include-stubs`: Include `.pyi` files when scanning a directory (single-file `.pyi` targets are analyzed as given)
- `--no-cache`: Disable the persistent on-disk embedding cache for this run
- `--strict-revision-cache`: Key an unpinned hub model's cache revision to a resolved commit hash instead of the requested revision label, disabling caching when a branch/tag can't be mapped offline (default: key by the requested label; a branch move is detected whenever a run loads the model, purging that shard so two checkpoints never mix, while fully warm runs keep serving the pre-move vectors coherently; see [Embedding cache](caching.md#what-invalidates-what))
- `--output-width <int>`: Rich render width for non-JSON output (default `160`, min `80`)
- `--json`: Emit JSON instead of rich tables
- `-v, --verbose`: Verbose logs

## `codedupes info`

Print version and runtime versions, supported languages and exact Tree-sitter package readiness, the effective default model and pinned revision, the built-in alias table with each profile's per-language semantic duplicate gates and search threshold, analysis defaults (minimum statements, exclude globs, output width), resolved device capabilities, MPS memory statistics when available, whether MLX is already loaded in the process, CPU identity with its capability-gated bfloat16 verdict and the effective CPU bfloat16 inference policy (native ISA, mkldnn availability, and the experimental `CODEDUPES_CPU_BF16=1` opt-in, see [Accelerators](accelerators.md#precision-and-metal-environment-variables)), and an embedding-cache summary (path, entry count, size on disk).

## `codedupes cache info`

Print the embedding-cache summary plus per-model entry counts and a per-repo breakdown.

## `codedupes cache clear [--model <name>]`

Clear all cached embeddings or only entries for one model. See [Embedding cache](caching.md).

## Validation and mode notes

- Threshold values must be in `[0.0, 1.0]`
- `--batch-size` and `--top-k` must be greater than `0`
- `--min-statements` and `--tiny-cutoff` must be greater than or equal to `0`
- `--output-width` must be at least `80`
- `--show-all` and `--allow-semantic-fallback` are only valid in default combined `check` mode (not with `--semantic-only` or `--traditional-only`)
- Default combined `check` fails if semantic backend fails; opt in to degraded combined fallback with `--allow-semantic-fallback`
- Missing/incompatible Tree-sitter grammars fail explicitly. Syntax recovery inside one source file is reported through extraction diagnostics, and affected units are skipped.
- Duplicate comparison is same-language by default; `--cross-language` opts semantic duplicate pairs into cross-language reporting. Semantic search always retrieves across every selected language.
- Unused-code analysis evaluates Python units only and reports the number of non-Python units excluded.
- A definition whose tokenized input (encode prompt included) exceeds the model's context window is skipped with a `semantic-context-overflow` diagnostic and the run continues; an over-long `search` query fails hard
- In `--json` mode, output is machine-parseable JSON only; warning text is surfaced via `summary.semantic_fallback` and `summary.semantic_fallback_reason` when fallback happens, and units the semantic stage skipped via the `semantic_diagnostics` array (`check` and `search` alike).
- Errors, parser-unavailable remediation, and warnings always go to stderr, so stdout stays parseable (a failed `--json` run writes nothing to stdout).
- `--json` rejects rich-only display controls: `--show-source`, `--full-table`, `--verbose`, and explicit `--output-width`
- `--semantic-only` and `--traditional-only` bypass hybrid synthesis and show raw method outputs
- `--semantic-only` and `--traditional-only` are mutually exclusive
- `--no-unused` and `--strict-unused` are mutually exclusive
- `--trust-remote-code` and `--no-trust-remote-code` are mutually exclusive
- `--mps-fallback` and `--no-mps-fallback` are mutually exclusive
- `--mps-memory-fraction` must be finite and in `(0, 2]`; it requires `--device mps` or `--device auto`
- Explicit semantic-analysis controls are rejected with `--traditional-only`, including model/task, candidate-scope, device/runtime options, and `--strict-revision-cache`. `--no-cache` is accepted as a harmless no-op.
- Explicit traditional-analysis controls are rejected with `--semantic-only`: `--traditional-threshold`, `--no-tiny-filter`, `--tiny-cutoff`, and `--tiny-near-jaccard-min`
- Unsupported-op fallback and codedupes OOM recovery are separate policies; `--no-mps-fallback` does not disable OOM recovery
- `search` applies semantic threshold filtering before returning `top-k` matches; without an explicit `--threshold`/`--semantic-threshold` it uses the model profile search default (for example `0.50` for `gte-modernbert-base`), not the stricter duplicate threshold
- `search` reports how many units it embedded: an index emptied by `--min-statements`, `--semantic-unit-type`, `--language`, `--exclude`, or `--no-private` prints a stderr warning instead of a bare "No matches found", and `--json` always carries the count in `indexed_units`
- Contradictory mode-specific options are rejected at parse time for the selected workflow

Inspect effective model defaults with:

```bash
codedupes info
```

For JSON payloads and complete exit-code semantics, see [Output and exit codes](output.md).
