# Installation and runtime environment

## Install (GitHub source)

```bash
pip install "codedupes @ git+https://github.com/pszemraj/codedupes.git"
```

Requires Python 3.11 or newer and PyTorch `>=2.13.0,<3`.

## Local development (editable install)

```bash
git clone https://github.com/pszemraj/codedupes.git
cd codedupes
pip install -e ".[dev]"
```

Dependency bounds are declared in [pyproject.toml](../pyproject.toml) and installed with the package.

## Polyglot parser dependencies

A normal installation includes the pinned Tree-sitter parser packages from [pyproject.toml](../pyproject.toml). They provide precompiled grammars; scanning does not download or compile them.

See [Polyglot language support](polyglot-languages.md) for parser architecture, extraction scope, and error handling.

## Verify the installation

Print installed runtime versions, parser availability, and device diagnostics:

```bash
codedupes info
```

On Apple Silicon, use macOS 14.0+ and a PyTorch wheel built with MPS support. Check the `MPS built/available` status in the output.

Continue with [accelerator behavior](accelerators.md), [model profiles](model-profiles.md), or the [CLI reference](cli.md).
