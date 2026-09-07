# Installation and runtime environment

## Install (GitHub source)

Use Python 3.11 or newer, with Git available for the source install:

Install [PyTorch for your platform](https://pytorch.org/get-started/locally/) **before installing codedupes**, or pip may select an unwanted build. The required version range is `>=2.13.0,<3`.

```bash
python -m pip install "codedupes @ git+https://github.com/pszemraj/codedupes.git"
codedupes info
```

Other dependencies are declared in [pyproject.toml](../pyproject.toml) and installed with the package. You do not need to clone this repository to analyze your own code.

A GPU is optional. Semantic analysis automatically uses available CUDA, Apple Silicon MPS, or CPU hardware; CPU inference can be slower. On Apple Silicon, use macOS 14.0+ and a PyTorch wheel built with MPS support. `codedupes info --verbose` reports installed runtime versions, parser availability, and device diagnostics, including `MPS built/available`.

If `codedupes` is not found after installation, ensure your Python installation's scripts directory is on `PATH` and that you are using the same Python installation as the install command. If installation reports no matching PyTorch distribution, check the Python/platform compatibility and the required version above; inference fallback only applies after installation succeeds.

## First scan and model downloads

From the project you want to analyze:

```bash
codedupes check ./src
```

Replace `./src` with an existing source directory or file. The first semantic run downloads the default `Alibaba-NLP/gte-modernbert-base` model from Hugging Face; later runs can reuse the downloaded model and cached embeddings. Source is processed locally. Internet access is needed for model assets that are not already available; see [local models and offline use](model-profiles.md#local-model-directories-and-offline-use).

To verify extraction and deterministic matching without loading a model:

```bash
codedupes check ./src --traditional-only --no-unused
```

A completed check may exit `1` because it found duplicates. See the [README quick start](../README.md#quick-start) for interpreting your first result and [exit codes](output.md#exit-codes) for automation.

## Polyglot parser dependencies

A normal installation includes the pinned Tree-sitter parser packages from [pyproject.toml](../pyproject.toml). They provide precompiled grammars; scanning does not download or compile them. You do not need a C, Rust, or JavaScript build toolchain to scan those source files.

See [Polyglot language support](polyglot-languages.md) for supported extensions, extraction scope, and parser errors.

## Local development (editable install)

To work on codedupes itself, clone the repository and install its development dependencies:

```bash
git clone https://github.com/pszemraj/codedupes.git
cd codedupes
python -m pip install -e ".[dev]"
```

Run the ordinary test suite without live accelerator or network tests, then check formatting and lint:

```bash
python -m pytest -m "not gpu and not mps and not network"
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

An unfiltered `python -m pytest` also runs the real CUDA or MPS tests when that hardware is available. Those tests load actual models and exercise memory exhaustion and recovery; they are not substitutes for the ordinary suite and may need downloaded model assets. Network smoke tests are opt-in:

```bash
CODEDUPES_SMOKE_NETWORK=1 python -m pytest tests/test_semantic_smoke.py -k network_smoke
```

For the real accelerator suites and model/search smoke checks, see [accelerator validation](accelerators.md#hardware-validation).

The source lives in `src/codedupes/`: `cli/` handles commands and reports, `analyzer.py` coordinates analysis, `extractor.py` and `languages/` extract definitions, and `traditional.py` / `semantic.py` implement matching. Tests live in `tests/`; labeled examples and calibration inputs live in `test_fixtures/`, with sweep tools in `scripts/`. Start with [analysis defaults](analysis-defaults.md) when changing behavior, or the [Python API](python-api.md) when integrating the package.
