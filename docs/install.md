# Installation and Runtime Environment

This page is the source of truth for installation and dependency/runtime environment setup.
Analysis behavior defaults are defined in
[docs/analysis-defaults.md](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md).
Semantic model-profile defaults are defined in
[docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md).

## Install (GitHub source)

```bash
pip install "codedupes @ git+https://github.com/pszemraj/codedupes.git"
```

Optional GPU extras:

```bash
pip install "codedupes[gpu] @ git+https://github.com/pszemraj/codedupes.git"
```

Requires Python 3.11+.

## Local development (editable install)

```bash
git clone https://github.com/pszemraj/codedupes.git
cd codedupes
pip install -e ".[dev]"
codedupes info
```

## Semantic dependency bounds

```bash
pip install "transformers>=4.51,<5" "sentence-transformers>=5,<6"
```

For C2LLM-family models, install `deepspeed` (via `codedupes[gpu]` or direct install).

For semantic model aliases/default thresholds/task behavior, see
[docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md).
For CLI flags (including `--model-revision` and `--trust-remote-code`), see
[docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md).
For runtime failure/exit behavior, see
[docs/output.md](https://github.com/pszemraj/codedupes/blob/main/docs/output.md).
