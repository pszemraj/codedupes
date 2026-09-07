# codedupes

`codedupes` finds duplicate code in Python, C, Rust, JavaScript/JSX, and TypeScript/TSX. It combines structural/token matching with semantic embeddings to find both similar syntax and similar intent. It also reports potentially unused Python definitions for review.

Use `check` to review duplicate candidates with file and line locations, or `search` to find functions and methods using a natural-language query. It analyzes source files without building or running your project.

## Install

Requires **Python 3.11+** and Git for installation from source:

Install [PyTorch for your platform](https://pytorch.org/get-started/locally/) **before installing codedupes**; otherwise pip may select a build you don't want. This repo requires PyTorch `>=2.13.0,<3`.

```bash
python -m pip install "codedupes @ git+https://github.com/pszemraj/codedupes.git"
codedupes info
```

The installation includes the supported language parsers. A GPU is optional: semantic inference automatically selects CUDA, Apple Silicon MPS, or CPU. See [installation](docs/install.md) for runtime requirements and editable development setup.

## Quick start

Run these from the repository you want to analyze. Replace `./src` with your source directory, `.` for the current directory, or a single supported source file.

```bash
# Find duplicates using both structural/token and semantic matching.
codedupes check ./src

# Find code by describing what it does.
codedupes search ./src "normalize request payload" --top-k 5

# Rank matching files, with a short list of contributing definitions.
codedupes search ./src "normalize request payload" --result-level file --top-k 5
```

The first semantic run downloads the default `gte-modernbert-base` model from Hugging Face and computes embeddings. It can take longer than later runs, which reuse cached embeddings. The scan runs locally; source code is not sent to a hosted embedding API. [Model profiles and offline use](docs/model-profiles.md) explain model selection and downloads.

For a first scan without loading or downloading an embedding model:

```bash
codedupes check ./src --traditional-only --no-unused
```

This checks structural/token duplicates and disables unused-code guesses. It still uses the same installed package and parser dependencies.

### Read the results

`check` prints duplicate locations, similarity evidence, and a summary. In combined mode, start with the `exact`, `traditional_near`, and `hybrid_confirmed` tiers; `semantic_high_confidence` and `semantic_review` are semantic candidates that need closer inspection. Scores rank candidates; they are not probabilities that two definitions are interchangeable. Potentially unused Python findings are also review suggestions, especially for callbacks and framework hooks.

**Exit code `1` can mean the scan found actionable duplicates, not that it crashed.** Runtime errors also use `1` and explain the failure on stderr; invalid options use `2`. For an interactive review that should succeed regardless of findings, or a machine-readable report:

```bash
codedupes check ./src --fail-on none
codedupes check ./src --json --fail-on none
```

See [output and exit codes](docs/output.md) for tiers, JSON fields, diagnostics, and CI policies.

Search returns code units by default. With `--result-level file`, each file appears once, ranked by its best matching unit, with up to three matching definitions and line numbers. `--top-k` then limits files. See [search options](docs/cli.md#codedupes-search-path-query) for scope and threshold controls.

### Know what gets scanned

- Tests and common dependency/build directories are excluded by default. Use `--no-default-excludes` to include tests, or repeat `--exclude` to narrow the scan.
- Semantic checks and search consider functions and methods with at least three statements by default. Traditional matching also considers classes and filters pairs where both definitions are tiny.
- Duplicate comparisons stay within each language by default. `--cross-language` opts into semantic comparisons across languages.
- Parsing is syntax-only: no compiler preprocessing, macro expansion, or project-wide name resolution. Unused-code analysis supports Python only.

If expected code is missing, check [analysis scope and filters](docs/analysis-defaults.md) and [supported files and parser limits](docs/polyglot-languages.md).

## Next steps

| Goal | Documentation |
| --- | --- |
| Adjust scan scope, thresholds, or search | [CLI reference](docs/cli.md) (`codedupes check --help` / `codedupes search --help`) |
| Call the analyzer from Python | [Python API](docs/python-api.md) |
| Interpret findings or integrate with CI | [Output and exit codes](docs/output.md) |
| Understand defaults and unused-code heuristics | [Analysis defaults](docs/analysis-defaults.md) |
| Check language support and extraction boundaries | [Polyglot language support](docs/polyglot-languages.md) |
| Change models or work offline | [Model profiles](docs/model-profiles.md) |
| Diagnose CPU, CUDA, or Apple Silicon execution | [Accelerators](docs/accelerators.md) |
| Inspect or clear cached embeddings | [Embedding cache](docs/caching.md) |
| Develop and test this project | [Development setup](docs/install.md#local-development-editable-install) |
| Reproduce calibration experiments | [Hybrid gate tuning](docs/hybrid-tuning.md) |
| Review upcoming behavior changes | [Next release changes](docs/release-notes-next.md) |
