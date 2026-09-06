# Analysis defaults and heuristics

These defaults apply to `codedupes check` and `AnalyzerConfig`. See the [CLI reference](cli.md) for syntax, [model profiles](model-profiles.md) for semantic thresholds and tasks, and [accelerators](accelerators.md) for device behavior.

## Semantic duplicate gate defaults

Semantic duplicate detection is gated per language: each built-in model profile carries a calibrated cosine gate for every supported language, measured against `test_fixtures/polyglot_calibration/`.

| language | `gte-modernbert-base` | `embeddinggemma-300m` |
| --- | --- | --- |
| python | `0.80` | `0.74` |
| c | `0.82` | `0.78` |
| rust | `0.74` | `0.78` |
| javascript | `0.70` | `0.72` |
| typescript | `0.68` | `0.78` |

Gate selection is recall-first. A shipped gate may sit below the sweep's F1-selected threshold wherever the sweep shows recall gains below it, however many grid steps down that is (gte `c` `0.82` against a selected `0.90`; embeddinggemma `javascript` `0.72` against `0.82` and `rust` `0.78` against `0.82`). Where recall is flat, a gate sits at most one grid step looser as an off-corpus generalization hedge, never further. Every shipped gate keeps recall at or above the selection's and F1 within 80% of it; `tests/test_calibration_reports.py` enforces both against the recorded sweep reports.

The profile fallback (`0.82` gte, `0.78` gemma) is the strictest calibrated gate and applies only to languages without their own entry. An explicit `--semantic-threshold`/`--threshold` (or `AnalyzerConfig.semantic_threshold`) replaces every per-language gate with one flat value. The pairwise embedding scan partitions candidates by language and scans each group at that language's own gate, so a loosely gated language never drags another language's scan down; the scalar floor handed to the scan covers only languages that arrive without a calibrated entry.

Semantic duplicate pairs are same-language by default. `--cross-language` (or `AnalyzerConfig(cross_language=True)`) also reports cross-language pairs; those claims are uncalibrated, so an opted-in mixed pair is held to `min(gate_a, gate_b)`, the looser of its two language gates.

Custom prompts, revisions, and trust settings must meet the [model profile threshold requirements](model-profiles.md#semantic-task-defaults-and-choices).

## Semantic candidate defaults

Default semantic candidate selection:

- unit types: `function`, `method`
- class units are excluded by default from semantic embedding
- minimum statement count: `3` (via `min_semantic_statements`)
- statements are counted recursively through control-flow bodies, so a large function implemented inside one outer block is not measured as a single statement; nested function/class definitions count as one declaration each. Python counts via the AST (`try`, `with`, loops, conditionals, `match`, with indented definitions dedented before counting); Tree-sitter languages apply each grammar's equivalent statement and nested-scope node rules, including Rust's semicolon-free tail expression as one statement
- each semantic input is one complete logical definition - signature, docstring, and body, starting at the definition line (`def`/`class` in Python; decorators are not included); functions are not split into arbitrary text chunks
- a definition whose tokenized input (the encode prompt included) exceeds the selected model's context window is never embedded from a partial prefix: it is skipped with a warning and a `semantic-context-overflow` diagnostic, and the run continues without it. `--allow-semantic-fallback` is unrelated to this path. An over-long `search` query still fails hard, because a truncated query has no result to omit

Traditional/semantic scope rule:

- traditional duplicate matching always uses the full extraction scope (functions, methods, and classes), in both combined and traditional-only modes
- semantic candidate controls such as `min_semantic_statements` and `semantic_unit_types` affect embeddings only; they cannot hide deterministic findings

Use the [CLI candidate options](cli.md#semantic-model) or `AnalyzerConfig.semantic_unit_types` and `min_semantic_statements` to change this selection.

## Extraction scope defaults

Directory-name exclusions always apply. They cover common artifact, dependency, and cache directories such as `node_modules`, `target`, `.venv`, `.pytest_cache`, `dist`, and `build`; directories ending in `.egg-info` are also skipped. A literal `vendor/` directory is not excluded by default: what the walk analyzes, the C-header policy scan also sees.

By default, these test-file globs apply:

- `**/test_*`
- `**/*_test.*`
- `**/*_tests.*`
- `**/*.test.*`
- `**/*.spec.*`
- `**/tests/**`
- `**/__tests__/**`

CLI `--exclude` options extend these patterns. Use `--no-default-excludes` to scan tests while retaining custom exclusions. For Python callers, `AnalyzerConfig.exclude_patterns=None` uses the defaults; a supplied list replaces them, including `[]` to disable test-file exclusions. Built-in artifact-directory exclusions always apply.

Bare names and basename globs match at any depth: `--exclude examples` skips both `examples/demo.py` and `pkg/examples/nested/demo.py`, without matching `myexamples`. A matched directory excludes all descendants and is pruned from traversal. A trailing `/` restricts a pattern to directories. Paths containing `/` match relative to the scan root; `./examples/` restricts the match to the root-level directory, while `**/examples/**` matches at any depth, including the root. Shell-style `*`, `?`, and character classes are supported; in path patterns `*` can also span `/`. Quote glob arguments in the shell.

Exclusions apply to direct file extraction too, relative to the file's parent for a single-file CLI target. Excluded symlink names are skipped before deduplication; aliases cannot reintroduce excluded in-tree targets.

Automatic C-header detection uses the same exclusions, so excluded C/C++ files do not affect whether included `.h` files are parsed as C.

## Potentially unused defaults

Unused detection evaluates Python units only; non-Python units are excluded and surfaced as a count (`unused_excluded_units`). It runs by default and builds a conservative reference graph from direct calls in analyzed code, module-level import and assignment aliases, `if __name__ == "__main__"` blocks, and `[project.scripts]` or `[project.gui-scripts]` entries in `pyproject.toml`.

The following units are not reported:

- referenced units (any analyzed call resolving to the unit's name or a qualified-name suffix counts)
- names exported through `__all__`, public classes, and dunder methods such as `__init__`
- `get_*` and `set_*` definitions of any unit type (not only methods - a module-level `get_thing()` is suppressed too, even in strict mode)
- definitions decorated with `@abstractmethod` or `@abc.abstractmethod`
- `test_*` definitions and definitions in files whose names contain `_test`
- units containing `# noqa: codedupes` or `# codedupes: ignore`

Call matching is name-based rather than scope-resolved: a call to any same-named symbol keeps every candidate definition out of the report, trading missed dead code for fewer false "unused" flags. Default mode also skips public non-method functions. Strict mode (`--strict-unused` or `strict_unused=True`) removes only that suppression; the other API and runtime exclusions still apply. Only call expressions count as references: attribute access without a call, decorator usage, callbacks passed as arguments, and type annotations do not, so framework-dispatched methods (for example `ast.NodeVisitor` `visit_*` hooks) surface as candidates. Dynamic registration, reflection, and string-based lookups likewise remain outside the static reference graph, so unused findings require review.

When unused detection runs, semantic duplicate pairs whose two units are both reported as potentially unused are removed before hybrid synthesis. Traditional duplicate findings remain available. `--no-unused` disables this suppression along with unused reporting.

## Tiny traditional duplicate filtering defaults

Default tiny-filter behavior for traditional duplicates:

- enabled: `True`
- tiny definition: effective code-unit statement count `< 3`; classes expand each extracted member from its declaration count to the member's statement count, so a class with a few substantial methods is not treated as a marker. JavaScript/TypeScript static initializer bodies are counted during extraction, including statements nested in control flow; an empty block still counts as one member. When private-unit filtering is active, class duplicates remain visible because their emitted member inventory may be incomplete
- traditional pairs where both units are tiny: dropped

Use `--no-tiny-filter` / `--tiny-cutoff`, or `AnalyzerConfig.filter_tiny_traditional` / `tiny_unit_statement_cutoff`, to change the filter.

## Hybrid synthesis confidence defaults

- semantic evidence: the per-language duplicate gate above (applied before synthesis; there is no separate semantic-only minimum)
- weak identifier jaccard minimum: `0.20`
- statement ratio minimum: `0.35`

A semantic-only pair retained after the [unused-code filter](#potentially-unused-defaults) has already passed its language's duplicate gate, so it remains visible in default output. Identifier overlap and a comparable statement count promote it to `semantic_high_confidence`; otherwise it is labeled `semantic_review`. These corroborators affect ranking and review priority, not admission. Tune them with the [hybrid gate workflow](hybrid-tuning.md).

## Confidence scale

Confidence combines similarity and corroborating evidence into a ranking score. Interpret it alongside the tier:

| tier | confidence |
| --- | --- |
| `exact` | `1.0` |
| `traditional_near` | `0.55 + 0.45 * jaccard` |
| `hybrid_confirmed` | `0.5 * semantic + 0.5 * jaccard` |
| `semantic_high_confidence` | `0.45 + 0.55 * semantic` |
| `semantic_review` | `0.40 + 0.45 * semantic` |

At the same semantic similarity, `semantic_review` scores below `semantic_high_confidence` by `0.05 + 0.10 * semantic`. Scores from different tiers can still overlap when their input similarities differ. Ties break on semantic similarity, then Jaccard, then unit uid.
