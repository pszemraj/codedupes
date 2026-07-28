# Installation and Runtime Environment

This page is the source of truth for installation and dependency/runtime environment setup.
Analysis behavior defaults are defined in
[docs/analysis-defaults.md](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md).
Semantic model-profile defaults are defined in
[docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md).
Accelerator requirements and runtime behavior are defined in
[docs/accelerators.md](https://github.com/pszemraj/codedupes/blob/main/docs/accelerators.md).

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

Use the standard install on macOS 14.0+ and verify that the installed PyTorch
wheel was built with MPS support:

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

For fallback, allocator, OOM-recovery, precision, and MLX guidance, see
[docs/accelerators.md](https://github.com/pszemraj/codedupes/blob/main/docs/accelerators.md).

For semantic model aliases/default thresholds/task behavior, see
[docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md).
For CLI flags (including `--model-revision` and `--trust-remote-code`), see
[docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md).
For runtime failure/exit behavior, see
[docs/output.md](https://github.com/pszemraj/codedupes/blob/main/docs/output.md).
