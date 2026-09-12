# Analysis defaults and heuristics

These defaults apply to `codedupes check` and `AnalyzerConfig` in check mode. See the [CLI reference](cli.md) for syntax, [model profiles](model-profiles.md) for semantic thresholds and tasks, and [accelerators](accelerators.md) for device behavior.

## What a default check does

`codedupes check <path>` runs combined duplicate detection: deterministic matching across every extracted function, method, and class, plus semantic comparison of eligible functions and methods. It also reports potentially unused Python units. Supported files, default test exclusions, and parser diagnostics determine the extracted set; see [polyglot language support](polyglot-languages.md#supported-files) and [extraction scope defaults](#extraction-scope-defaults).

The semantic pass may load the selected embedding model and may download it on its first use. [CLI options](cli.md#codedupes-check-path) cover single-method and unused-analysis controls.

Combined output assigns each pair an evidence tier and sorts by [confidence](#confidence-scale):

| tier | evidence |
| --- | --- |
| `exact` | structural or token fingerprints agree |
| `traditional_near` | identifier Jaccard match |
| `hybrid_confirmed` | semantic and traditional-near match |
| `semantic_high_confidence` | semantic match plus size/identifier corroboration or a calibrated similarity margin |
| `semantic_review` | semantic match only |

See [report selection](output.md#report-selection) and [exit codes](output.md#exit-codes) for visibility and failure-policy behavior.

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

See [threshold-profile choices](model-profiles.md#choosing-threshold-defaults) for profile selection.

The profile fallback (`0.82` gte, `0.78` gemma) is the strictest calibrated gate and applies only to languages without their own entry. An explicit `--semantic-threshold`/`--threshold` (or `AnalyzerConfig.semantic_threshold`) replaces every per-language gate with one flat value. The pairwise embedding scan partitions candidates by language and scans each group at that language's own gate, so a loosely gated language never drags another language's scan down; the scalar floor handed to the scan covers only languages that arrive without a calibrated entry.

Semantic duplicate pairs are same-language by default. `--cross-language` (or `AnalyzerConfig(cross_language=True)`) also reports cross-language pairs; those claims are uncalibrated, so an opted-in mixed pair is held to `min(gate_a, gate_b)`, the looser of its two language gates.

Custom prompts, revisions, and trust settings must meet the [model profile threshold requirements](model-profiles.md#semantic-task-defaults-and-choices).

## Semantic candidate defaults

Default semantic candidate selection:

- unit types: `function`, `method`
- class units are excluded by default from semantic embedding
- minimum statement count: `3` (via `min_semantic_statements`)
- statements are counted recursively through control-flow bodies, so a large function implemented inside one outer block is not measured as a single statement; nested function/class definitions count as one declaration each. Each grammar defines its statement and nested-scope node kinds: Python follows `ast.stmt` semantics (`elif` counts, a `with` statement counts once plus its body, only `else`/`except`/`finally`/`case` clauses are transparent, a leading docstring is not counted; see [Python units](polyglot-languages.md#python)), and Rust's semicolon-free tail expression counts as one statement. Every backend sets the count at extraction.
- each semantic input is one complete logical definition - the unit's exact source span of decorators (a decorated Python definition starts at its first decorator), signature, docstring, and body; functions are not split into arbitrary text chunks
- eligible definitions and search queries are passed to the embedding backend unchanged. The backend applies its normal tokenization and context-window truncation, including any encode prompt.

When a newly encoded unit exceeds the loaded model's context window, `semantic_diagnostics` includes a `semantic-context-overflow` warning with its token count and source location. The unit remains searchable and eligible for duplicate detection. Counts include the encode prompt and special tokens. Cache-only runs do not load a tokenizer just to repeat warnings; use `--no-cache` to recheck every selected unit. This diagnostic covers corpus units, not query length.

Traditional/semantic scope rule:

- traditional duplicate matching always uses the full extraction scope (functions, methods, and classes), in both combined and traditional-only modes
- semantic candidate controls such as `min_semantic_statements` and `semantic_unit_types` affect embeddings only; they cannot hide deterministic findings

Use the [CLI candidate options](cli.md#semantic-model) or `AnalyzerConfig.semantic_unit_types` and `min_semantic_statements` to change this selection.

## Extraction scope defaults

Directory-name exclusions prune directories beneath the scan root. They cover common artifact, dependency, and cache directories such as `node_modules`, `target`, `.venv`, `.pytest_cache`, `dist`, and `build`; directories ending in `.egg-info` are also skipped. The selected root and its ancestors are outside exclusion matching: selecting `node_modules/` directly scans its contents. A literal `vendor/` directory is not excluded by default: what the walk analyzes, the C-header policy scan also sees.

By default, these test-file globs apply:

- `**/test_*`
- `**/*_test.*`
- `**/*_tests.*`
- `**/*.test.*`
- `**/*.spec.*`
- `**/tests/**`
- `**/__tests__/**`

CLI `--exclude` options extend these patterns for directory scans. Scans do not read `.gitignore`; add explicit exclusions such as `--exclude scratch` for other local checkouts or generated sources. Use `--no-default-excludes` to scan tests while retaining custom exclusions. For Python callers, `AnalyzerConfig.exclude_patterns=None` uses the defaults; a supplied list replaces them, including `[]` to disable test-file exclusions. An explicitly named source file bypasses the default test-file patterns, but supplied `--exclude` options and `AnalyzerConfig.exclude_patterns` still apply relative to its parent directory. That parent becomes the scan root, so its own name and ancestor names do not exclude the file. Built-in artifact-directory exclusions beneath the scan root remain active.

Directory scans log an INFO hint when default test patterns skip files or prune directories. The counts cover encountered source files and pruned directories; they do not enumerate files inside those directories. `--json` suppresses this informational log.

Bare names and basename globs match at any depth: `--exclude examples` skips both `examples/demo.py` and `pkg/examples/nested/demo.py`, without matching `myexamples`. A matched directory excludes all descendants and is pruned from traversal. A trailing `/` restricts a pattern to directories. Paths containing `/` match relative to the scan root; `./examples/` restricts the match to the root-level directory, while `**/examples/**` matches at any depth, including the root. Shell-style `*`, `?`, and character classes are supported; in path patterns `*` can also span `/`. Quote glob arguments in the shell.

Custom exclusions apply to direct file extraction too, relative to the file's parent for a single-file CLI target. `check` and `search` preserve an explicitly named file symlink for exclusion matching. Excluded symlink names are skipped before deduplication; aliases cannot reintroduce excluded in-tree targets. Targets outside the scan root retain the symlink's in-tree name for extraction and exclusions.

[C-header detection](polyglot-languages.md#c-headers) uses the same exclusions and symlink identity rules as extraction.

## Potentially unused defaults

Unused detection evaluates Python units only; non-Python units are excluded and surfaced as a count (`unused_excluded_units`). It runs by default. `--no-unused` (`run_unused=False`) disables it without changing duplicate findings; `--strict-unused` (`strict_unused=True`) also reports unreferenced public functions and public methods. Every Python file the extractor visits is parsed once with the standard-library `ast`, whether or not it yielded units, so a re-export module or a script still contributes references; no model or grammar is loaded. A file `ast` rejects (a syntax error the grammar recovered from, syntax newer than the interpreter, an expression nested past the recursion limit) contributes no references and logs a warning naming it, so units only that file references may surface as unused.

A unit is referenced when module-level code or another definition uses its name through any of:

- a loaded name: calls, decorators, base classes, default arguments, callbacks passed as values, and class-body aliases such as `visit_Name = _impl`
- attribute access in any context, including stores through a property setter and bound-method callbacks such as `onerror=self._cleanup`
- annotations on parameters, returns, and annotated assignments, including quoted forward references such as `"Node | None"`; the string values of `Literal[...]` are not names
- `import` and `from ... import` statements, which reference what they import
- module-level statements, `if __name__ == "__main__"` blocks included (module code runs on import)
- `[project.scripts]`, `[project.gui-scripts]`, and `[project.entry-points]` group entries in `pyproject.toml`
- framework dispatch: public methods of a class whose base does not resolve by name to a class in the analyzed tree (`object` excluded; subclasses of such a class inherit the rule), such as `ast.NodeVisitor` `visit_*` hooks or `logging.Filter.filter`

Module-level import and assignment aliases (`from .mod import _Props as _Base`, `_Alias = _Later`) expand a reference to its target; an assignment alias is not itself a reference, and the framework rule resolves bases through the same aliases. Matching is name-based rather than scope-resolved: a reference to `helper` keeps every unit named `helper` (or whose qualified name ends in the referenced dotted path) out of the report, trading missed dead code for fewer false "unused" flags. Likewise an external base whose last segment matches a project class name resolves as project. A unit's own body never counts for itself, methods included, so a self-recursive helper nobody calls is still reported; references inside a nested definition count for every enclosing definition, except a nested definition's reference to itself. Dynamic registration, reflection, and string lookups such as `getattr(obj, "name")` stay outside the graph, so unused findings require review.

Unreferenced units are still not reported when they are:

- names exported through `__all__`, public classes, and dunder methods such as `__init__`
- `get_*` and `set_*` definitions of any unit type (not only methods - a module-level `get_thing()` is suppressed too, even in strict mode)
- definitions whose own decorators include `@abstractmethod` or `@abc.abstractmethod` (the enclosing class and a body that merely mentions the text are not exempt)
- `test_*` definitions and definitions in files whose names contain `_test`
- units containing `# noqa: codedupes` or `# codedupes: ignore`

Default mode also skips public surface: a function or method every segment of whose qualified name is public (`pkg.mod.run`, `Service.run`) is API, not a finding. A public name reached only through a private module, class, or function (`_Service.run`, `_factory.helper`, `_factory.Local.run`) and every private definition stay reportable. Dunder module names such as `__main__` also fail the public-surface rule: unreferenced functions in an entry-point script are reportable by default, while calls from its main block still mark their targets as referenced. Strict mode removes only that suppression; the exclusions above still apply, and framework-dispatched methods stay referenced because that rule is a reference, not a policy.

Unused findings are independent of duplicate detection: a potentially unused unit remains eligible for semantic and traditional duplicate reporting.

## Traditional duplicate defaults

The traditional pass reports near-duplicate pairs when identifier-set Jaccard similarity is at least `0.85` by default (`jaccard_threshold`). Structural and token exact matches do not use this threshold. [Fingerprint and comparison boundaries](polyglot-languages.md#fingerprints-and-comparison-boundaries) define which units can be paired.

Default tiny-filter behavior for traditional duplicates:

- enabled: `True`
- tiny definition: effective code-unit statement count `< 3`; classes expand each extracted member from its declaration count to the member's statement count, so a class with a few substantial methods is not treated as a marker. JavaScript/TypeScript static initializer bodies are counted during extraction, including statements nested in control flow; an empty block still counts as one member. When private-unit filtering is active, class duplicates remain visible because their emitted member inventory may be incomplete
- traditional pairs where both units are tiny: dropped

Use `--no-tiny-filter` / `--tiny-cutoff`, or `AnalyzerConfig.filter_tiny_traditional` / `tiny_unit_statement_cutoff`, to change the filter.

## Hybrid synthesis confidence defaults

A semantic-only pair has already passed its language's duplicate gate (applied before synthesis; there is no separate semantic-only minimum). Synthesis then splits it into `semantic_high_confidence` or `semantic_review`; the split affects ranking and default visibility, never admission. A pair is promoted when either path holds:

- corroboration: weak identifier Jaccard >= `hybrid_weak_identifier_jaccard_min` and statement-count ratio >= `hybrid_statement_ratio_min`, both on the model profile;
- similarity promotion: cosine >= the language's `language_high_confidence_thresholds` entry on the profile (cross-language pairs must clear the stricter of the two gates; a language without a calibrated entry has promotion off).

| profile | identifier Jaccard min | statement ratio min | promotion gates |
| --- | --- | --- | --- |
| `gte-modernbert-base` | `0.00` | `0.80` | typescript `0.88`; off elsewhere |
| `embeddinggemma-300m` | `0.00` | `0.20` | off |
| `generic` | `0.00` | `0.20` | off |

Both corroboration constants are one pooled selection per profile from the [corroboration sweep](hybrid-tuning.md#run-the-sweep) at the shipped admission gates. Identifier overlap is measured from the same identifier collection in every language (Python's includes attribute and keyword-argument names, see [fingerprints](polyglot-languages.md#fingerprints-and-comparison-boundaries)), so the identifier floor is a cross-language measurement, not an artifact of one extractor. It is `0.00` because the semantic-only positives in every corpus are alpha-renamed: at each profile's shipped statement-ratio floor no positive identifier floor is feasible in all five languages - even `0.05` drops Rust and Python below recall retention under both models, TypeScript below recall retention under `gte-modernbert-base` and below published precision under `embeddinggemma-300m`, and C below published precision under `embeddinggemma-300m`. The statement-ratio floor carries GTE's split; EmbeddingGemma's `0.20` floor withholds only extreme size mismatches. An explicit `--semantic-threshold` keeps the profile's corroboration constants but turns promotion off because the gates are calibrated relative to the shipped admission gates. See the [calibration results](../test_fixtures/polyglot_calibration/README.md#calibration-results) and [hybrid gate workflow](hybrid-tuning.md).

## Confidence scale

Finite cosine scores are bounded to [-1, 1] before reporting, so float32 rounding cannot produce values above 1. Confidence combines similarity and corroborating evidence into a ranking score. Interpret it alongside the tier:

| tier | confidence |
| --- | --- |
| `exact` | `1.0` |
| `traditional_near` | `0.55 + 0.45 * jaccard` |
| `hybrid_confirmed` | `0.5 * semantic + 0.5 * jaccard` |
| `semantic_high_confidence` | `0.45 + 0.55 * semantic` |
| `semantic_review` | `0.40 + 0.45 * semantic` |

At the same semantic similarity, `semantic_review` scores below `semantic_high_confidence` by `0.05 + 0.10 * semantic`. Scores from different tiers can still overlap when their input similarities differ. Ties break on semantic similarity, then Jaccard, then unit uid.
