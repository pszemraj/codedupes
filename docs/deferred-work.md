# Deferred work

Capabilities this project deliberately does not have. Each entry records what it would be, why it is absent, and where it would land, so the question is answered once instead of re-litigated per review. Nothing here is committed work.

## Configuration file (`[tool.codedupes]`)

Every setting comes from a CLI flag or a hard-coded default; there is no project-level configuration. A `[tool.codedupes]` table in `pyproject.toml` is the obvious home, and discovery is already solved — `unused.find_pyproject` locates the nearest `pyproject.toml` at or above the scan target.

Precedence is the part that is not solved. `cli/_options.py` distinguishes an explicitly passed flag from its default (`_is_cli_explicit`) and rejects flag combinations on that basis; a third source between flag and default turns each of those checks into a three-way question, and mode flags such as `--unused-only` would have to reject configured values the same way they reject explicit ones. Until a workflow needs it, a shell alias or a CI step carries the same flags with none of that.

## Machine-readable `info` and `cache info`

`codedupes check` and `codedupes search` emit JSON; `codedupes info` (`cli/info.py`) and `codedupes cache info` (`cli/cache.py`) print Rich tables only, so automation has to scrape them or import the package.

Adding `--json` to both is small in isolation. The reason to wait is overlap: a check report's `run` record already carries the resolved model, revision, profile, requested and executed device, and the per-language gates, leaving environment probing and cache statistics as the only unique payload. Those deserve their own schemas rather than a second copy of `run`.

## Framework-derived methods under `--strict-unused`

`--strict-unused` reports public functions and public methods, and the reference graph is name-based over the project's own source. A method that exists because a framework calls it — a `Model.save` override, a pytest plugin hook, a callback registered by decorator, a subclass filling in a base-class contract — has no in-repo caller and is reported.

Today's exemptions are narrow and syntactic: `abstractmethod` (`unused._is_abstract`), dunder and `__init__`/`__new__`/`__call__` names, `__all__` exports, public classes (`CodeUnit.is_likely_api`), `get_`/`set_` prefixes, pyproject entry points, and anything carrying a `codedupes: ignore[unused]` directive. Exempting framework overrides means either resolving base classes across files — real inheritance analysis, not a suffix index — or shipping a per-framework list of decorators and base classes that goes stale between releases. A directive at the definition states the same fact and stays checkable:

```text
def save(self, *args, **kwargs):  # codedupes: ignore[unused] ORM hook
```

## Unit-type counts in JSON

The terminal Analysis Summary breaks `Total code units` down by language and by Functions / Methods / Classes (`cli/_render.py`). JSON carries `run.units.extracted` and `run.units.semantic_eligible` only, and `units[]` holds just the units referenced by reported findings, so a consumer cannot reconstruct the breakdown from a report.

Per-language and per-unit-type counts under `run.units` would close the gap, and `_build_run_record` already receives the unit list, so the computation is a `Counter` over `unit.language` and `unit.unit_type`. It is deferred for scope, not difficulty — a JSON addition also carries the schema, `docs/output.md`, and consumer-facing consequences that belong in a change of their own.
