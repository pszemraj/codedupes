# Analysis Defaults and Heuristics

These defaults apply to `codedupes check` and `AnalyzerConfig`. See the [CLI reference](cli.md) for syntax, [model profiles](model-profiles.md) for semantic thresholds and tasks, and [accelerators](accelerators.md) for device behavior.

## Semantic Duplicate Gate Defaults

Semantic duplicate detection is gated per language: each built-in model profile carries a calibrated cosine gate for every supported language, measured against `test_fixtures/polyglot_calibration/` with recall-leaning selection (the loosest threshold whose F1 stays near that language's best).

| language | `gte-modernbert-base` | `embeddinggemma-300m` |
|---|---|---|
| python | `0.80` | `0.74` |
| c | `0.82` | `0.78` |
| rust | `0.74` | `0.78` |
| javascript | `0.70` | `0.72` |
| typescript | `0.68` | `0.76` |

The profile fallback (`0.82` gte, `0.78` gemma) is the strictest calibrated gate and applies only to languages without their own entry. An explicit `--semantic-threshold`/`--threshold` (or `AnalyzerConfig.semantic_threshold`) replaces every per-language gate with one flat value. The pairwise embedding scan runs at the loosest gate among the languages present, then each pair is held to its own language's gate.

Semantic duplicate pairs are same-language by default. `--cross-language` (or `AnalyzerConfig(cross_language=True)`) also reports cross-language pairs; those claims are uncalibrated, and an opted-in mixed pair is held to the looser of its two language gates.

## Semantic Candidate Defaults

Default semantic candidate selection:

- unit types: `function`, `method`
- class units are excluded by default from semantic embedding
- minimum statement count: `3` (via `min_semantic_statements`)
- statements are counted recursively through control-flow bodies, so a large function implemented inside one outer block is not measured as a single statement; nested function/class definitions count as one declaration each. Python counts via the AST (`try`, `with`, loops, conditionals, `match`, with indented definitions dedented before counting); Tree-sitter languages apply each grammar's equivalent statement and nested-scope node rules
- each semantic input is one complete logical definition - signature, docstring, and body, starting at the definition line (`def`/`class` in Python; decorators are not included); functions are not split into arbitrary text chunks

Combined-mode alignment rule:

- when both traditional and semantic analysis are enabled, traditional duplicate matching is scoped to the same semantic candidate pool
- traditional-only mode keeps full extraction scope (functions, methods, classes)

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

Directory-name exclusions always apply. They cover common artifact, vendor, and cache directories such as `node_modules`, `target`, `.venv`, `.pytest_cache`, `dist`, and `build`; directories ending in `.egg-info` are also skipped.

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
- tiny definition: function/method statement count `< 3`
- tiny exact duplicates: dropped
- tiny near duplicates: kept only when Jaccard similarity `>= 0.93`

Override via CLI:

```bash
codedupes check ./src --no-tiny-filter
codedupes check ./src --tiny-cutoff 4 --tiny-near-jaccard-min 0.95
```

Override via Python API:

```python
AnalyzerConfig(
    filter_tiny_traditional=False,
    tiny_unit_statement_cutoff=4,
    tiny_near_jaccard_min=0.95,
)
```

## Hybrid Synthesis Gate Defaults

- semantic evidence: the per-language duplicate gate above (applied before synthesis; there is no separate semantic-only minimum)
- weak identifier jaccard minimum: `0.20`
- statement ratio minimum: `0.35`

A gated semantic-only pair must additionally pass the identifier-overlap and statement-ratio guards to reach the `semantic_high_confidence` tier. Tune the guards with the [hybrid gate workflow](hybrid-tuning.md).
