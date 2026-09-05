# Analysis Defaults and Heuristics

These defaults apply to `codedupes check` and `AnalyzerConfig`. See the [CLI reference](cli.md) for syntax, [model profiles](model-profiles.md) for semantic thresholds and tasks, and [accelerators](accelerators.md) for device behavior.

## Semantic Duplicate Gate Defaults

Semantic duplicate detection is gated per language: each built-in model profile carries a calibrated cosine gate for every supported language, measured against `test_fixtures/polyglot_calibration/`.

| language | `gte-modernbert-base` | `embeddinggemma-300m` |
|---|---|---|
| python | `0.80` | `0.74` |
| c | `0.82` | `0.78` |
| rust | `0.74` | `0.78` |
| javascript | `0.70` | `0.72` |
| typescript | `0.68` | `0.78` |

Gate selection is recall-first. A shipped gate may sit below the sweep's F1-selected threshold wherever the sweep shows recall gains below it, however many grid steps down that is (gte `c` `0.82` against a selected `0.90`; embeddinggemma `javascript` `0.72` against `0.82` and `rust` `0.78` against `0.82`). Where recall is flat, a gate sits at most one grid step looser as an off-corpus generalization hedge, never further. Every shipped gate keeps recall at or above the selection's and F1 within 80% of it; `tests/test_calibration_reports.py` enforces both against the recorded sweep reports.

The profile fallback (`0.82` gte, `0.78` gemma) is the strictest calibrated gate and applies only to languages without their own entry. An explicit `--semantic-threshold`/`--threshold` (or `AnalyzerConfig.semantic_threshold`) replaces every per-language gate with one flat value. The pairwise embedding scan partitions candidates by language and scans each group at that language's own gate, so a loosely gated language never drags another language's scan down; the scalar floor handed to the scan covers only languages that arrive without a calibrated entry.

Semantic duplicate pairs are same-language by default. `--cross-language` (or `AnalyzerConfig(cross_language=True)`) also reports cross-language pairs; those claims are uncalibrated, so an opted-in mixed pair is held to `min(gate_a, gate_b)`, the looser of its two language gates.

Default duplicate gates are calibrated for the profile's pinned revision, default task, default prompt, and default remote-code setting. A custom instruction prefix, an alternate EmbeddingGemma task, an alternate built-in revision, or a `trust_remote_code` value differing from the profile default is uncalibrated context: the run refuses the default gates and requires an explicit threshold.

## Semantic Candidate Defaults

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

Override via CLI:

```bash
codedupes check ./src --semantic-unit-type class
codedupes check ./src --min-statements 0
```

Override via Python API:

```python
AnalyzerConfig(
    semantic_unit_types=("function", "method", "class"),
    min_semantic_statements=0,
)
```

## Extraction Scope Defaults

Directory-name exclusions always apply. They cover common artifact, dependency, and cache directories such as `node_modules`, `target`, `.venv`, `.pytest_cache`, `dist`, and `build`; directories ending in `.egg-info` are also skipped. A literal `vendor/` directory is not excluded by default: what the walk analyzes, the C-header policy scan also sees.

When no nonempty `exclude_patterns` list is supplied, these file globs apply:

- `**/test_*`
- `**/*_test.*`
- `**/*_tests.*`
- `**/*.test.*`
- `**/*.spec.*`
- `**/tests/**`
- `**/__tests__/**`

A nonempty `AnalyzerConfig.exclude_patterns` list or one or more CLI `--exclude` options replaces those file globs. Directory-name exclusions still apply. Repeat any built-in file globs that you want to preserve alongside custom patterns.

## Potentially Unused Defaults

Unused detection evaluates Python units only; non-Python units are excluded and surfaced as a count (`unused_excluded_units`). It runs by default and builds a conservative reference graph from direct calls in analyzed code, module-level import and assignment aliases, `if __name__ == "__main__"` blocks, and `[project.scripts]` or `[project.gui-scripts]` entries in `pyproject.toml`.

The following units are not reported:

- referenced units (any analyzed call resolving to the unit's name or a qualified-name suffix counts)
- names exported through `__all__`, public classes, and dunder methods such as `__init__`
- `get_*` and `set_*` definitions of any unit type (not only methods - a module-level `get_thing()` is suppressed too, even in strict mode)
- `test_*` definitions and definitions in files whose names contain `_test`
- units containing `# noqa: codedupes` or `# codedupes: ignore`

Call matching is name-based rather than scope-resolved: a call to any same-named symbol keeps every candidate definition out of the report, trading missed dead code for fewer false "unused" flags. Default mode also skips public non-method functions. Strict mode (`--strict-unused` or `strict_unused=True`) removes only that suppression; the other API and runtime exclusions still apply. Only call expressions count as references: attribute access without a call, decorator usage, callbacks passed as arguments, and type annotations do not, so framework-dispatched methods (for example `ast.NodeVisitor` `visit_*` hooks) surface as candidates. Dynamic registration, reflection, and string-based lookups likewise remain outside the static reference graph, so unused findings require review.

## Tiny Traditional Duplicate Filtering Defaults

Default tiny-filter behavior for traditional duplicates:

- enabled: `True`
- tiny definition: effective code-unit statement count `< 3`; classes expand each extracted member from its declaration count to the member's statement count, so a class with a few substantial methods is not treated as a marker. JavaScript/TypeScript static initializer bodies are counted during extraction, including statements nested in control flow; an empty block still counts as one member. When private-unit filtering is active, class duplicates remain visible because their emitted member inventory may be incomplete
- traditional pairs where both units are tiny: dropped

Override via CLI:

```bash
codedupes check ./src --no-tiny-filter
codedupes check ./src --tiny-cutoff 4
```

Override via Python API:

```python
AnalyzerConfig(
    filter_tiny_traditional=False,
    tiny_unit_statement_cutoff=4,
)
```

## Hybrid Synthesis Confidence Defaults

- semantic evidence: the per-language duplicate gate above (applied before synthesis; there is no separate semantic-only minimum)
- weak identifier jaccard minimum: `0.20`
- statement ratio minimum: `0.35`

A semantic-only pair has already passed its language's duplicate gate, so it remains visible in default output. Identifier overlap and a comparable statement count promote it to `semantic_high_confidence`; otherwise it is labeled `semantic_review`. These corroborators affect ranking and review priority, not admission. Tune them with the [hybrid gate workflow](hybrid-tuning.md).

## Confidence Scale

Confidence is a corroboration scale, not a raw similarity, so a tier with more independent evidence always outranks one with less at equal evidence strength:

| tier | confidence |
|---|---|
| `exact` | `1.0` |
| `traditional_near` | `0.55 + 0.45 * jaccard` |
| `hybrid_confirmed` | `0.5 * semantic + 0.5 * jaccard` |
| `semantic_high_confidence` | `0.45 + 0.55 * semantic` |
| `semantic_review` | `0.40 + 0.45 * semantic` |

The last two formulas keep `semantic_review` strictly below `semantic_high_confidence` at every similarity (the gap is `0.05 + 0.10 * semantic`), so uncorroborated pairs can never crowd corroborated ones off the top of the table. Ties break on semantic similarity, then Jaccard, then unit uid.
