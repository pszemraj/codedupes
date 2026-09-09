# Installation and runtime environment

## Install (GitHub source)

Use Python 3.11 or newer, with Git available for the source install:

Install [PyTorch for your platform](https://pytorch.org/get-started/locally/) **before installing codedupes**, or pip may select an unwanted build. The required version range is `>=2.13.0,<3`.

```bash
pip install "codedupes @ git+https://github.com/pszemraj/codedupes.git"
codedupes info
```

Other dependencies are declared in [pyproject.toml](../pyproject.toml) and installed with the package. You do not need to clone this repository to analyze your own code.

A GPU is optional. On Apple Silicon, use macOS 14.0+ and a PyTorch wheel built with MPS support. [Accelerator behavior](accelerators.md) covers automatic device selection and diagnostics.

If `codedupes` is not found after installation, ensure your Python installation's scripts directory is on `PATH` and that you are using the same Python installation as the install command. If installation reports no matching PyTorch distribution, check the Python/platform compatibility and the required version above; inference fallback only applies after installation succeeds. Continue with the [README quick start](../README.md#quick-start).

## Polyglot parser dependencies

A normal installation includes the pinned Tree-sitter parser packages from [pyproject.toml](../pyproject.toml). They provide precompiled grammars; scanning does not download or compile them. You do not need a C, Rust, or JavaScript build toolchain to scan those source files.

See [Polyglot language support](polyglot-languages.md) for supported extensions, extraction scope, and parser errors.

## Local development (editable install)

To work on codedupes itself, clone the repository and install its development dependencies:

```bash
git clone https://github.com/pszemraj/codedupes.git
cd codedupes
pip install -e ".[dev]"
```

Run the ordinary test suite without live accelerator or network tests, then check formatting and lint:

```bash
pytest -m "not gpu and not mps and not network"
ruff check src tests scripts
ruff format --check src tests scripts
```

An unfiltered `pytest` also runs the real CUDA or MPS tests when that hardware is available. Those tests load actual models and exercise memory exhaustion and recovery; they are not substitutes for the ordinary suite and may need downloaded model assets. Network smoke tests are opt-in:

```bash
CODEDUPES_SMOKE_NETWORK=1 pytest tests/test_semantic_smoke.py -k network_smoke
```

For the real accelerator suites and model/search smoke checks, see [accelerator validation](accelerators.md#hardware-validation).

The source lives in `src/codedupes/`: `cli/` handles commands and reports, `analyzer.py` coordinates analysis, `extractor.py` and `languages/` extract definitions, and `traditional.py` / `semantic.py` implement matching. Tests live in `tests/`; labeled examples and calibration inputs live in `test_fixtures/`, with sweep tools in `scripts/`. Start with [analysis defaults](analysis-defaults.md) when changing behavior, or the [Python API](python-api.md) when integrating the package.
