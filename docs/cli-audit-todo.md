# CLI audit TODO

Scope: issue [#7](https://github.com/pszemraj/codedupes/issues/7), related extraction surprises, and adversarial checks of the installed CLI. Ordered by impact and dependency; each fix includes regression coverage.

1. [x] Reject an explicitly empty `cache clear --model` value: it previously cleared every model, treating an empty scope as an omitted scope. Regression tests preserve a populated cache for empty and whitespace-only values.
2. [ ] Fix exclusion semantics for `check` and `search`: bare names match at any depth, matched directories exclude descendants and are pruned, and custom CLI patterns extend the test defaults. Provide `--no-default-excludes` for intentionally scanning tests; keep built-in artifact exclusions. Make Python `exclude_patterns=[]` mean no default file patterns.
3. [ ] Audit exclusion boundaries: root-relative paths, trailing slashes, globs, near-miss names, direct files, nested directories, and symlinks. Apply exclusions before symlink deduplication so excluded aliases cannot hide included targets.
4. [ ] Make automatic C-header detection honor the same exclusions: ignored C/C++ sources must not change the language interpretation of included headers.
5. [ ] Reject contextual search without an explicit threshold before indexing, with usage status 2. Normalize search construction and missing-file failures through the existing CLI error handler.
6. [ ] Exercise help/version, invalid numbers and paths, option conflicts and environment precedence, JSON stdout/stderr, empty results, finding exit policies, and cache scope preservation. Update existing CLI/default documentation and run the relevant tests, lint, formatting, and the broader local suite.

Boundary: no new detection algorithms, dependency changes, accelerator execution, or release infrastructure. Existing Click interrupt behavior and documented successful JSON stderr suppression remain intentional. Model inference is represented by existing test doubles for CLI regressions.
