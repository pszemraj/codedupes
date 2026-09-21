# codedupes

`codedupes` finds duplicate code in Python, C, Rust, JavaScript/JSX, and TypeScript/TSX. It combines structural/token matching with semantic embeddings to find both similar syntax and similar intent. It also reports potentially unused Python definitions, ones nothing in the tree references, for review.

Use `check` to review duplicate candidates with file and line locations, or `search` to find functions and methods using a natural-language query. It analyzes source files without building or running your project.

## Install

```bash
pip install "codedupes @ git+https://github.com/pszemraj/codedupes.git"
codedupes info
```

See [installation](docs/install.md) for prerequisites, parser/runtime requirements, and editable development setup.

## Quick start

Run these from the repository you want to analyze. Replace `./src` with your source directory, `.` for the current directory, or a single supported source file.

```bash
# Find duplicates using both structural/token and semantic matching.
codedupes check ./src

# Find code by describing what it does.
codedupes search ./src "normalize request payload" --top-k 5
```

See the [CLI reference](docs/cli.md) for file-level ranking and [model profiles](docs/model-profiles.md) before the first semantic run.

For a first scan without loading or downloading an embedding model:

```bash
codedupes check ./src --traditional-only --no-unused
```

This checks structural/token duplicates and disables unused-code guesses. It still uses the same installed package and parser dependencies.

See [output and exit codes](docs/output.md) for findings and CI behavior. [Analysis scope and filters](docs/analysis-defaults.md) and [language support](docs/polyglot-languages.md) define what is scanned.

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
