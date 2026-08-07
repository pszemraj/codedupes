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
