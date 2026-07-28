# Analysis Defaults and Heuristics

These defaults apply to `codedupes check` and `AnalyzerConfig`. See the
[CLI reference](cli.md) for syntax, [model profiles](model-profiles.md) for semantic thresholds and
tasks, and [accelerators](accelerators.md) for device behavior.

## Semantic Candidate Defaults

Default semantic candidate selection:

- unit types: `function`, `method`
- class units are excluded by default from semantic embedding
- minimum statement count: `3` (via `min_semantic_lines`)

Combined-mode alignment rule:

- when both traditional and semantic analysis are enabled, traditional duplicate
  matching is scoped to the same semantic candidate pool
- traditional-only mode keeps full extraction scope (functions, methods, classes)

Override via CLI:

```bash
codedupes check ./src --semantic-unit-type class
codedupes check ./src --min-lines 0
```

Override via Python API:

```python
AnalyzerConfig(
    semantic_unit_types=("function", "method", "class"),
    min_semantic_lines=0,
)
```

## Extraction Scope Defaults

Default extraction excludes are always applied unless code is changed:

- directory names: common artifact/vendor/cache directories (for example
  `node_modules`, `target`, `.venv`, `.pytest_cache`, `dist`, `build`)
- file globs:
  - `**/test_*`
  - `**/*_test.py`
  - `**/tests/**`

CLI `--exclude` adds patterns on top of these defaults; it does not replace
them.

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

- semantic-only minimum: `0.92`
- weak identifier jaccard minimum: `0.20`
- statement ratio minimum: `0.35`

Tune these values with the [hybrid gate workflow](hybrid-tuning.md).
