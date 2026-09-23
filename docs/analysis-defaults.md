# Analysis defaults and heuristics

These defaults apply to `codedupes check` and `AnalyzerConfig` in check mode. See the [CLI reference](cli.md) for syntax, [model profiles](model-profiles.md) for semantic thresholds and tasks, and [accelerators](accelerators.md) for device behavior.

## What a default check does

`codedupes check <path>` runs combined duplicate detection: deterministic matching across every extracted function, method, and class, plus semantic comparison of eligible functions and methods. It also reports potentially unused Python units. Supported files, default test exclusions, and parser diagnostics determine the extracted set; see [polyglot language support](polyglot-languages.md#supported-files) and [extraction scope defaults](#extraction-scope-defaults).

The semantic pass may load the selected embedding model and may download it on its first use. [CLI options](cli.md#codedupes-check-path) cover single-method and unused-analysis controls.

Combined output assigns each pair an evidence tier and sorts by [score](#score-scale); reports then list actionable tiers before advisory ones, see [report selection](output.md#report-selection):

| tier | evidence |
| --- | --- |
| `exact` | structural or token fingerprints agree; reported as [families](output.md#exact-families), one record per set of copies |
| `traditional_near` | identifier Jaccard match |
| `hybrid_confirmed` | semantic and traditional-near match |
| `semantic_high_confidence` | semantic match plus size/identifier corroboration or a calibrated similarity margin |
| `semantic_review` | semantic match only |

See [report selection](output.md#report-selection) and [exit codes](output.md#exit-codes) for visibility and failure-policy behavior.

## Semantic duplicate gate defaults

Semantic duplicate detection is gated per language. [Built-in duplicate and search gates](model-profiles.md#duplicate-and-search-gates) are profile policy; the [calibration workflow](hybrid-tuning.md) records their development evidence.

An explicit `--semantic-threshold`/`--threshold` (or `AnalyzerConfig.semantic_threshold`) replaces every per-language gate with one flat value. The pairwise embedding scan partitions candidates by language, so a loosely gated language never drags another language's scan down.

Semantic duplicate pairs are same-language by default. `--cross-language` (or `AnalyzerConfig(cross_language=True)`) also reports cross-language pairs; those claims are uncalibrated, so an opted-in mixed pair is held to `min(gate_a, gate_b)`, the looser of its two language gates.

Custom prompts, revisions, and trust settings must meet the [model profile threshold requirements](model-profiles.md#semantic-task-defaults-and-choices).

## Semantic candidate defaults

Default semantic candidate selection:

- unit types: `function`, `method`
- class units are excluded by default from semantic embedding
- minimum statement count: `3` (via `min_semantic_statements`); each backend computes recursive counts from its grammar, so a function implemented inside one outer control-flow block is not treated as one statement. See [language extraction details](polyglot-languages.md#what-becomes-a-code-unit).
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

Directory scans inside a git work tree also skip whatever git ignores. The ignored set comes from `git ls-files --others --ignored --exclude-standard`, so nested `.gitignore` files, `.git/info/exclude`, the global excludes file, negations, and tracked files that match a pattern all behave exactly as they do for git; an ignored directory is pruned without listing its contents. Pass `--no-gitignore` (`AnalyzerConfig.respect_gitignore=False`) to scan ignored paths. Outside a work tree, or when `git` is not on `PATH`, `.gitignore` files are plain files and nothing is skipped. Selecting an ignored directory or file directly scans it: the rules gate discovery beneath the root, not the root itself.

CLI `--exclude` options extend these patterns for directory scans. Use `--no-default-excludes` to scan tests while retaining custom exclusions. For Python callers, `AnalyzerConfig.exclude_patterns=None` uses the defaults; a supplied list replaces them, including `[]` to disable test-file exclusions. An explicitly supplied list of the same default patterns is still a user exclusion for unused-code references. An explicitly named source file bypasses the default test-file patterns, but supplied `--exclude` options and `AnalyzerConfig.exclude_patterns` still apply relative to its parent directory. That parent becomes the scan root, so its own name and ancestor names do not exclude the file. Built-in artifact-directory exclusions beneath the scan root remain active.

A single-file target still extracts duplicate candidates from that file alone. When it yields Python units, its [unused-code references](#potentially-unused-defaults) come from the whole project: the nearest `pyproject.toml` at or above the file (falling back to the git work-tree root, then the file's own directory) becomes a reference root, walked with the same custom `--exclude`/git-ignore/artifact-directory rules but without the default test-file shapes, so a call from a test file elsewhere in the project still counts. Explicitly anchored custom exclusions (`./` or `/`) remain relative to the selected file's parent during this wider reference walk; unanchored path and basename patterns still apply elsewhere in the project.

Directory scans log an INFO hint when default test patterns skip files or prune directories, and another when git ignore rules do. The counts cover encountered source files and pruned directories; they do not enumerate files inside those directories. `--json` suppresses these informational logs.

A file or directory skipped only by a default test-file shape (not a real `--exclude` pattern, a git ignore rule, or an artifact directory) is still parsed for [unused-code references](#potentially-unused-defaults): its units are absent from duplicate detection, but a call inside `def test_helper(): ...` still credits the production code it exercises. Directory symlinks are not followed by either walk. Add the shape to `--exclude`, even when it is identical to a default pattern, to drop a test file from references too; a real ancestor directory name such as `tests` works as well.

Bare names and basename globs match at any depth: `--exclude examples` skips both `examples/demo.py` and `pkg/examples/nested/demo.py`, without matching `myexamples`. A matched directory excludes all descendants and is pruned from traversal. A trailing `/` restricts a pattern to directories. Paths containing `/` match relative to the scan root; `./examples/` restricts the match to the root-level directory, while `**/examples/**` matches at any depth, including the root. Shell-style `*`, `?`, and character classes are supported; in path patterns `*` can also span `/`. Quote glob arguments in the shell.

Custom exclusions apply to direct file extraction too, relative to the file's parent for a single-file CLI target. `check` and `search` preserve an explicitly named file symlink for exclusion matching. Excluded symlink names are skipped before deduplication; aliases cannot reintroduce excluded in-tree targets. Targets outside the scan root retain the symlink's in-tree name for extraction and exclusions.

[C-header detection](polyglot-languages.md#c-headers) uses the same exclusions and symlink identity rules as extraction.

## Suppression directives

A comment containing `codedupes: ignore` attaches to the code unit it precedes or trails, recorded on `CodeUnit.suppressions`. The grammar is one form, comments only, and case-sensitive: `codedupes: ignore` alone names every kind; `codedupes: ignore[unused]`, `codedupes: ignore[duplicates]`, and `codedupes: ignore[unused, duplicates]` narrow it to specific kinds, with free text after the closing bracket read as a reason and ignored (`codedupes: ignore[unused] # flagged for removal next sprint`). A kind list must stay on the directive's line; a later line of a block comment is not part of it. Matching is exact: `ignored` and `noqa: codedupes` do not match. An unclosed kind list produces a `suppression-syntax` diagnostic and suppresses nothing. An unrecognized kind name inside a closed list produces a `suppression-syntax` diagnostic naming it, while any recognized kinds in the same directive still apply.

A directive attaches to a unit two ways: trailing within its header - the `def`/signature line, a decorator or attribute line, between decorators, the opening brace of the body on its own row (`{ // codedupes: ignore`), or the end of a one-line unit (`def f(): return 1  # codedupes: ignore`) - or as a contiguous block of own-line comments directly above it, with no blank line breaking the run. A comment that shares a row with other code (`x = 1  # codedupes: ignore`) is not a leading comment for whatever follows it, and one trailing a statement with more of the body after it belongs to the body, not the unit. A Rust `#[attribute]` between a doc comment and the item it decorates is transparent to the search, so a doc comment above `#[inline]` still attaches. A directive applies to its own unit and to every unit nested inside it - a class-level directive marks every method - but never the reverse: a directive above a nested definition marks only that definition, not its enclosing one. A directive inside a string or docstring is not a comment and never attaches.

## Potentially unused defaults

Unused detection evaluates Python units only; non-Python units are excluded and surfaced as a count (`unused_excluded_units`). It runs by default in checks. `--no-unused` (`run_unused=False`) disables it without changing duplicate findings or walking files solely for the unused reference graph; `CodeAnalyzer.index()` likewise skips reference-only file discovery regardless of this check setting. `--strict-unused` (`strict_unused=True`) also reports unreferenced public functions and public methods. `--unused-only` runs unused detection alone, with both duplicate methods disabled and no embedding model loaded; see [`check --unused-only`](cli.md#codedupes-check-path). Every Python file the extractor visits is parsed once with the standard-library `ast`, whether or not it yielded units, so a re-export module or a script still contributes references; the same walk also covers files [skipped only by a default test-file shape](#extraction-scope-defaults), directory scan and single-file target alike, so a call inside a default-excluded test still credits what it calls. In-tree file symlink aliases share one source identity in this reference graph, so the same function body cannot credit itself through an alias; no model or grammar is loaded. A directory traversal failure during either reference walk produces an extraction `walk-error` and marks the run partial. The interpreter recursion limit is raised for the reference walk only (never lowered) so a generated elif or operator chain a few hundred deep, well inside what tree-sitter extracts without complaint, is analyzed rather than bailed out of. A file that cannot be read, that `ast` rejects (a syntax error the grammar recovered from, syntax newer than the interpreter), or whose nesting overflows even the raised recursion limit (a pathological chain, or `ast.parse` itself exhausting its C stack) contributes no references; the analysis logs a warning and records one diagnostic per file (`unused-read-error`, `unused-parse-error`, `unused-recursion-limit`) in `unused_diagnostics`, so units only that file references may surface as unused.

A unit is referenced when module-level code or another definition uses its name through any of:

- a loaded name: calls, decorators, base classes, default arguments, callbacks passed as values, and class-body aliases such as `visit_Name = _impl`
- attribute access in any context, including stores through a property setter and bound-method callbacks such as `onerror=self._cleanup`
- annotations on parameters, returns, and annotated assignments, including quoted forward references such as `"Node | None"`; the string values of `Literal[...]` are not names
- `import` and `from ... import` statements, which reference what they import
- module-level statements, `if __name__ == "__main__"` blocks included (module code runs on import)
- `[project.scripts]`, `[project.gui-scripts]`, and `[project.entry-points]` group entries in the nearest `pyproject.toml` at or above the scan target (a directory root or a single file's own directory), stopping after the git work-tree root so an unrelated ancestor project is never picked up; each target credits only its full `module:object` path, and a bare module or value missing `:` credits nothing
- framework dispatch: public methods of a class whose base does not resolve by name to a class in the analyzed tree (`object` excluded; subclasses of such a class inherit the rule), such as `ast.NodeVisitor` `visit_*` hooks or `logging.Filter.filter`

Module-level import and assignment aliases (`from .mod import _Props as _Base`, `_Alias = _Later`) expand a reference to its target; an assignment alias is not itself a reference, and the framework rule resolves bases through the same aliases. Matching is mostly name-based: a reference to `helper` keeps every unit named `helper` (or whose qualified name ends in the referenced dotted path) out of the report, trading missed dead code for fewer false "unused" flags. A bare name loaded inside a definition resolves to the same module's definitions of that name, in the nearest enclosing function that has one or else at the top level, instead of to every same-named unit, so each copy of a copied module credits only its own helpers; statement order is ignored. Class-body names, attribute names, module-level code, and names the module does not define still use name-based matching. A definition the extractor did not turn into a reported unit (filtered by `--no-private`, nested in a filtered-out private container, or located in a default-excluded test) still credits names it loads outside its own lexical definitions, so what it calls is not falsely reported as unused. Likewise an external base whose last segment matches a project class name resolves as project. A unit's own body never counts for itself, methods included, so a self-recursive helper nobody calls is still reported; references inside a nested definition count for every enclosing definition, except a nested definition's reference to itself. Dynamic registration, reflection, and string lookups such as `getattr(obj, "name")` stay outside the graph, so unused findings require review.

Unreferenced units are still not reported when they are:

- names exported through `__all__`, public classes, and dunder methods such as `__init__`
- `get_*` and `set_*` definitions of any unit type (not only methods - a module-level `get_thing()` is suppressed too, even in strict mode)
- definitions with a decorator line matching `@abstractmethod` or `@abc.abstractmethod` exactly (the enclosing class, a body that merely mentions the text, and a same-prefix decorator name such as `@abstractmethodish` are not exempt)
- `test_*` definitions and definitions in files matching the default test-file exclude shapes: a `test_*` file-name prefix, or a `_test`/`_tests` file-name stem suffix (`legacy_testament.py` does not match; `probe_test.py`, `probe_tests.py`, and `test_probe.py` do)
- units carrying a `codedupes: ignore` or `codedupes: ignore[unused]` suppression directive (see [Suppression directives](#suppression-directives) above); the retired `# noqa: codedupes` marker and the old substring check over unit source no longer suppress anything

Default mode also skips public surface: a function or method every segment of whose qualified name is public (`pkg.mod.run`, `Service.run`) is API, not a finding. A public name reached only through a private module, class, or function (`_Service.run`, `_factory.helper`, `_factory.Local.run`) and every private definition stay reportable. Dunder module names such as `__main__` also fail the public-surface rule: unreferenced functions in an entry-point script are reportable by default, while calls from its main block still mark their targets as referenced. Strict mode removes only that suppression; the exclusions above still apply, and framework-dispatched methods stay referenced because that rule is a reference, not a policy.

Unused findings are independent of duplicate detection: a potentially unused unit remains eligible for semantic and traditional duplicate reporting. A directive-suppressed unit that would otherwise have been a finding is counted in `suppressed_unused`, whether or not it carries `[duplicates]` too.

## Traditional duplicate defaults

The traditional pass reports near-duplicate pairs when identifier-set Jaccard similarity is at least `0.85` by default (`jaccard_threshold`). Structural and token exact matches do not use this threshold. [Fingerprint and comparison boundaries](polyglot-languages.md#fingerprints-and-comparison-boundaries) define which units can be paired.

Default tiny-filter behavior for traditional duplicates:

- enabled: `True`
- tiny definition: effective code-unit statement count `< 3`; classes expand each extracted member from its declaration count to the member's statement count, so a class with a few substantial methods is not treated as a marker. JavaScript/TypeScript static initializer bodies are counted during extraction, including statements nested in control flow; an empty block still counts as one member. When private-unit filtering is active, class duplicates remain visible because their emitted member inventory may be incomplete
- traditional pairs where both units are tiny: dropped

Use `--no-tiny-filter` / `--tiny-cutoff`, or `AnalyzerConfig.filter_tiny_traditional` / `tiny_unit_statement_cutoff`, to change the filter.

A `codedupes: ignore[duplicates]` (or bare `codedupes: ignore`) directive on either endpoint drops the pair entirely, after the tiny filter and before hybrid synthesis, so it never resurfaces in an exact family or a hybrid edge. Dropped pairs are counted in `suppressed_duplicates`; the same directive applies to semantic pairs, dropped after the `--suppress-test-semantic` filter on the same terms.

## Hybrid synthesis confidence defaults

A semantic-only pair has already passed its language's duplicate gate (applied before synthesis; there is no separate semantic-only minimum). Synthesis then splits it into `semantic_high_confidence` or `semantic_review`; the split affects ranking and default visibility, never admission. A pair is promoted by corroboration or a similarity gate:

- corroboration requires both the profile's weak identifier Jaccard and statement-count ratio;
- similarity promotion requires the profile's language gate (cross-language pairs must clear the stricter gate; a language without a calibrated entry has promotion off).

The [hybrid confidence gates](model-profiles.md#hybrid-confidence-gates) define the shipped values. An explicit `--semantic-threshold` keeps the profile's corroboration constants but turns similarity promotion off because those promotion gates belong to the shipped profile policy.

## Score scale

Finite cosine scores are bounded to [-1, 1] before reporting, so float32 rounding cannot produce values above 1. `score` combines similarity and corroborating evidence into a ranking key; it is not a calibrated probability — a `traditional_near` pair over identical identifier sets scores `1.0` even when the bodies differ by one operator. Interpret it alongside the tier:

| tier | score |
| --- | --- |
| `exact` | `1.0` |
| `traditional_near` | `0.55 + 0.45 * jaccard` |
| `hybrid_confirmed` | `0.5 * semantic + 0.5 * jaccard` |
| `semantic_high_confidence` | `0.45 + 0.55 * semantic` |
| `semantic_review` | `0.40 + 0.45 * semantic` |

At the same semantic similarity, `semantic_review` scores below `semantic_high_confidence` by `0.05 + 0.10 * semantic`. Scores from different tiers can still overlap when their input similarities differ. `exact` pairs sort ahead of every other tier, including near pairs that also reach `1.0` at perfect Jaccard; within the remaining tiers ties break on semantic similarity, then Jaccard, then unit uid.
