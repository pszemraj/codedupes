# Installation and Runtime Environment

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
codedupes info
```

Dependency bounds are declared in [pyproject.toml](../pyproject.toml) and installed with the package.

## Polyglot parser dependencies

A normal codedupes installation also installs the exact parser versions used by the extraction backends:

```text
tree-sitter==0.25.2
tree-sitter-c==0.24.2
tree-sitter-rust==0.24.2
tree-sitter-javascript==0.25.0
tree-sitter-typescript==0.23.2
```

These packages provide precompiled grammars; scanning does not download or compile them. Confirm the installed dialects with:

```bash
codedupes info
```

See [Polyglot language support](polyglot-languages.md) for parser architecture, extraction scope, and error handling.

## Apple Silicon / MPS

Use the standard install on macOS 14.0+ and verify that the installed PyTorch wheel was built with MPS support:

```bash
python - <<'PY'
import torch

print("torch", torch.__version__)
print("mps_built", torch.backends.mps.is_built())
print("mps_available", torch.backends.mps.is_available())
PY
```

Continue with [accelerator behavior](accelerators.md), [model profiles](model-profiles.md), or the [CLI reference](cli.md).
