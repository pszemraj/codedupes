# Installation and Runtime Environment

## Install (GitHub source)

```bash
pip install "codedupes @ git+https://github.com/pszemraj/codedupes.git"
```

Requires Python 3.11+ and PyTorch `>=2.13.0,<3`.

## Local development (editable install)

```bash
git clone https://github.com/pszemraj/codedupes.git
cd codedupes
pip install -e ".[dev]"
codedupes info
```

## Semantic dependency bounds

```bash
pip install "torch>=2.13.0,<3" "transformers>=5.1,<6" "sentence-transformers>=5.6,<6" "numpy>=2.1.0,<3"
```

## Polyglot parser dependencies

A normal codedupes installation also installs the exact parser versions used by the extraction backends:

```text
tree-sitter==0.25.2
tree-sitter-c==0.24.2
tree-sitter-rust==0.24.2
tree-sitter-javascript==0.25.0
tree-sitter-typescript==0.23.2
```

These packages provide precompiled grammars. codedupes does not download or compile a grammar while scanning a repository, and a missing/incompatible parser is an explicit error rather than a line-chunk fallback. Confirm the installed dialects with:

```bash
codedupes info
```

JavaScript and JSX share the JavaScript grammar. TypeScript and TSX use separate grammar entry points from the TypeScript package. See [Polyglot language support](polyglot-languages.md) for extraction scope and known limits.

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

Then run:

```bash
codedupes check ./src --device mps
```

Continue with [accelerator behavior](accelerators.md), [model profiles](model-profiles.md), or the [CLI reference](cli.md).
