# Semantic Model Profiles and Tasks

This document is the source of truth for semantic model-profile behavior:

- built-in model aliases and canonical IDs
- model-family runtime defaults (threshold, revision, trust mode)
- semantic task defaults and accepted task names

Installation/dependency guidance lives in
[docs/install.md](https://github.com/pszemraj/codedupes/blob/main/docs/install.md).
CLI option syntax/validation lives in
[docs/cli.md](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md).

The canonical implementation is:

- [`src/codedupes/semantic_profiles.py`](https://github.com/pszemraj/codedupes/blob/main/src/codedupes/semantic_profiles.py)
- [`src/codedupes/constants.py`](https://github.com/pszemraj/codedupes/blob/main/src/codedupes/constants.py)
- [`src/codedupes/semantic.py`](https://github.com/pszemraj/codedupes/blob/main/src/codedupes/semantic.py)

## Built-in profiles

| profile key | canonical model ID | family | default threshold | default revision | default trust mode |
| --- | --- | --- | --- | --- | --- |
| `gte-modernbert-base` | `Alibaba-NLP/gte-modernbert-base` | `gte-modernbert` | `0.96` | `auto` (unpinned) | `False` |
| `c2llm-0.5b` | `codefuse-ai/C2LLM-0.5B` | `c2llm` | `0.80` | `bd6d0ddb29f0c9a3d0f14281aedc9f940bb8d67a` | `True` |
| `embeddinggemma-300m` | `unsloth/embeddinggemma-300m` | `embeddinggemma` | `0.86` | `auto` (unpinned) | `False` |

Notes:

- C2LLM-family profiles require `deepspeed`.
- Generic/unknown models fall back to threshold `0.82` unless you override `--semantic-threshold` / `semantic_threshold`.

## Alias resolution rules

- Built-in alias keys and known aliases resolve to the profile’s canonical model ID.
- Any model name containing `c2llm` resolves to a dynamic C2LLM-family profile:
  - `trust_remote_code=True` by default
  - no pinned default revision
- Other unknown model IDs resolve to the generic profile.

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

- `c2llm`: uses family-specific instruction prefixes for code/query/describe modes.
- `embeddinggemma`: uses task-aware query/document prefix formats.
- generic models: no default instruction prefix unless explicitly overridden.

For usage examples, see
[docs/usage.md](https://github.com/pszemraj/codedupes/blob/main/docs/usage.md)
and
[docs/python-api.md](https://github.com/pszemraj/codedupes/blob/main/docs/python-api.md).
