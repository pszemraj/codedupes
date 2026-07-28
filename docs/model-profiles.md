# Semantic Model Profiles and Tasks

This document is the source of truth for semantic model-profile behavior:

- built-in model aliases and canonical IDs
- model-family runtime defaults (threshold, revision, trust mode)
- semantic task defaults and accepted task names

Installation/dependency guidance lives in
[docs/install.md](https://github.com/pszemraj/codedupes/blob/main/docs/install.md).
CLI option syntax/validation lives in
[docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md).
Device placement and MPS precision/memory policy live in
[docs/accelerators.md](https://github.com/pszemraj/codedupes/blob/main/docs/accelerators.md).

The canonical implementation is:

- [`src/codedupes/semantic_profiles.py`](https://github.com/pszemraj/codedupes/blob/main/src/codedupes/semantic_profiles.py)
- [`src/codedupes/constants.py`](https://github.com/pszemraj/codedupes/blob/main/src/codedupes/constants.py)
- [`src/codedupes/semantic.py`](https://github.com/pszemraj/codedupes/blob/main/src/codedupes/semantic.py)

## Built-in profiles

| profile key | canonical model ID | family | duplicate threshold | search threshold | default revision | default trust mode |
| --- | --- | --- | --- | --- | --- | --- |
| `gte-modernbert-base` | `Alibaba-NLP/gte-modernbert-base` | `gte-modernbert` | `0.96` | `0.50` | `auto` (unpinned) | `False` |
| `embeddinggemma-300m` | `unsloth/embeddinggemma-300m` | `embeddinggemma` | `0.86` | `0.40` | `auto` (unpinned) | `False` |

Notes:

- The duplicate threshold gates `check` pair reporting; the search threshold is the floor for
  `search` query matches. Query-to-code similarity runs far below code-to-code duplicate
  similarity, so search defaults are intentionally much lower.
- Search defaults are calibrated recall-safe: across seven real corpora, genuinely relevant
  hits start near `0.59` while fully off-topic queries ceiling near `0.48` on most corpora.
  Vocabulary overlap — a shared domain (GPU kernels vs a graphics query) or even a single
  shared word ("pattern", "parse", "warp") — can push off-topic matches to `0.52`–`0.65`;
  those carry visible scores and rank below real hits, but raise `--semantic-threshold`
  toward `0.6` if they clutter results. No fixed floor separates them everywhere, and the
  default deliberately favors recall over precision.
- Built-in EmbeddingGemma dtype overrides use float32 on MPS. CUDA may use
  bfloat16 when the hardware reports support.
- Generic/unknown model profiles follow their model-defined dtype and operator support, so explicit MPS compatibility is not guaranteed.
- Generic/unknown models fall back to duplicate threshold `0.82` and search threshold `0.35`
  unless you override `--semantic-threshold` / `semantic_threshold`.

## Alias resolution rules

- Built-in alias keys and known aliases resolve to the profile’s canonical model ID.
- A model name that is an existing directory on disk is treated as a local
  `save_pretrained`-style model copy: it canonicalizes to its resolved absolute
  path (so relative and absolute spellings share one identity), and its family is
  inferred from the directory basename using the same name rules below.
- Any model name containing `embeddinggemma` resolves to a dynamic
  EmbeddingGemma-family profile with the built-in profile’s thresholds and encode
  entry points.
- Any model name containing `gte-modernbert` resolves to a dynamic
  gte-modernbert-family profile with the built-in profile’s calibrated thresholds.
- Other unknown model IDs resolve to the generic profile.

### Local model directories and offline use

Pass a directory path anywhere a model name is accepted to load a model straight
from disk without contacting the HuggingFace Hub:

```bash
codedupes check ./src --model /opt/models/embeddinggemma-300m --device mps
```

- `--model-revision` is ignored for local directories (with a warning): on-disk
  weights have no hub revision. The embedding cache instead keys local models by a
  content fingerprint of the directory, so replacing or retraining the weights in
  place invalidates cached vectors automatically.
- For hub models already present in the local HuggingFace cache, set
  `HF_HUB_OFFLINE=1` to guarantee no network access.

For live effective values in your environment, run:

```bash
codedupes info
```

## Semantic task defaults and choices

Task defaults:

- `check`: `semantic-similarity`
- `search`: `code-retrieval`

Allowed task names:

- `semantic-similarity`
- `code-retrieval`
- `retrieval`
- `question-answering`
- `fact-verification`
- `classification`
- `clustering`

If you pass an unknown semantic task, the CLI/API raises a validation error.

## Task/prefix behavior by model family

- `embeddinggemma`: uses task-aware query/document prefix formats.
- generic models: no default instruction prefix unless explicitly overridden.

For usage examples, see
[docs/usage.md](https://github.com/pszemraj/codedupes/blob/main/docs/usage.md)
and
[docs/python-api.md](https://github.com/pszemraj/codedupes/blob/main/docs/python-api.md).
