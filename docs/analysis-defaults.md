# Analysis Defaults and Heuristics

This document is the source of truth for analysis-behavior defaults used by
`codedupes check` and the Python analyzer API.

For CLI command syntax and option validation, see
[docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md).
For model/runtime installation defaults, see
[docs/install.md](https://github.com/pszemraj/codedupes/blob/main/docs/install.md).

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

Semantic-only hybrid gate defaults are defined in
[`src/codedupes/analyzer.py`](https://github.com/pszemraj/codedupes/blob/main/src/codedupes/analyzer.py):

- semantic-only minimum: `0.92`
- weak identifier jaccard minimum: `0.20`
- statement ratio minimum: `0.35`

These values should be tuned using the workflow in
[docs/hybrid-tuning.md](https://github.com/pszemraj/codedupes/blob/main/docs/hybrid-tuning.md).
