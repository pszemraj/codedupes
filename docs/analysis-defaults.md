# Analysis Defaults and Heuristics

These defaults apply to `codedupes check` and `AnalyzerConfig`. See the [CLI reference](cli.md) for syntax, [model profiles](model-profiles.md) for semantic thresholds and tasks, and [accelerators](accelerators.md) for device behavior.

## Semantic Candidate Defaults

Default semantic candidate selection:

- unit types: `function`, `method`
- class units are excluded by default from semantic embedding
- minimum statement count: `3` (via `min_semantic_lines`)

Combined-mode alignment rule:

- when both traditional and semantic analysis are enabled, traditional duplicate matching is scoped to the same semantic candidate pool
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

Directory-name exclusions always apply. They cover common artifact, vendor, and cache directories such as `node_modules`, `target`, `.venv`, `.pytest_cache`, `dist`, and `build`; directories ending in `.egg-info` are also skipped.

When no nonempty `exclude_patterns` list is supplied, these file globs apply:

- `**/test_*`
- `**/*_test.py`
- `**/tests/**`

A nonempty `AnalyzerConfig.exclude_patterns` list or one or more CLI `--exclude` options replaces those file globs. Directory-name exclusions still apply. Repeat any built-in file globs that you want to preserve alongside custom patterns.

## Potentially Unused Defaults

Unused detection runs by default and builds a conservative reference graph from direct calls, module-level aliases, `if __name__ == "__main__"` blocks, and `[project.scripts]` or `[project.gui-scripts]` entries in `pyproject.toml`.

The following units are not reported:

- referenced units and proven `ast.NodeVisitor` or `ast.NodeTransformer` dispatch hooks (inheritance is proven through imports across the analyzed files, including relative imports; unresolvable third-party bases stay eligible for reporting)
- names exported through `__all__`, public classes, and dunder/API lifecycle methods such as `__init__`, `__new__`, and `__call__`
- `get_*`, `set_*`, and abstract methods
- `test_*` definitions and definitions in files whose names contain `_test`
- units containing `# noqa: codedupes` or `# codedupes: ignore`

Default mode also skips public top-level functions. Strict mode (`--strict-unused` or `strict_unused=True`) removes only that last suppression; the other API and runtime exclusions still apply. Dynamic registration and reflection remain outside the static call graph, so unused findings require review.

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
