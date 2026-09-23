# Deferred work

Capabilities this project deliberately does not have. Each entry records what it would be, why it is absent, and where it would land, so the question is answered once instead of re-litigated per review. Nothing here is committed work.

## Configuration file (`[tool.codedupes]`)

Every setting comes from a CLI flag or a hard-coded default; there is no project-level configuration. A `[tool.codedupes]` table in `pyproject.toml` is the obvious home, and discovery is already solved — `unused.find_pyproject` locates the nearest `pyproject.toml` at or above the scan target.

Precedence is the part that is not solved. `cli/_options.py` distinguishes an explicitly passed flag from its default (`_is_cli_explicit`) and rejects flag combinations on that basis; a third source between flag and default turns each of those checks into a three-way question, and mode flags such as `--unused-only` would have to reject configured values the same way they reject explicit ones. Until a workflow needs it, a shell alias or a CI step carries the same flags with none of that.

## Machine-readable `info` and `cache info`

`codedupes check` and `codedupes search` emit JSON; `codedupes info` (`cli/info.py`) and `codedupes cache info` (`cli/cache.py`) print Rich tables only, so automation has to scrape them or import the package.

Adding `--json` to both is small in isolation. The reason to wait is overlap: a check report's `run` record already carries the resolved model, revision, profile, requested and executed device, and the per-language gates, leaving environment probing and cache statistics as the only unique payload. Those deserve their own schemas rather than a second copy of `run`.
