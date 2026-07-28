# Semantic Model Profiles and Tasks

Profiles resolve model aliases, thresholds, revisions, trust settings, and task-specific embedding
behavior. See [Installation](install.md) for dependencies, the [CLI reference](cli.md) for option
syntax, and [Accelerators](accelerators.md) for device and precision behavior.

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
  Vocabulary overlap from a shared domain (GPU kernels vs a graphics query) or even a single
  shared word ("pattern", "parse", "warp") can push off-topic matches to `0.52`-`0.65`;
  those carry visible scores and rank below real hits, but raise `--semantic-threshold`
  toward `0.6` if they clutter results. No fixed floor separates them everywhere, and the
  default deliberately favors recall over precision.
- Generic/unknown models fall back to duplicate threshold `0.82` and search threshold `0.35`
  unless you override `--semantic-threshold` / `semantic_threshold`.

## Alias resolution rules

- Built-in alias keys and known aliases resolve to the profile's canonical model ID.
- A model name that is an existing directory on disk is treated as a local
  model copy before built-in aliases are considered, then canonicalized to its
  resolved absolute path — including the on-disk letter case on
  case-insensitive filesystems such as macOS — so relative, absolute, and
  differently-cased spellings share one cache identity.
- Known local model families are inferred from a recognizable directory name,
  Hugging Face cache ancestor, saved configuration, or model-card title.
  Unrecognized local models use the generic thresholds.
- Any model name containing `embeddinggemma` resolves to a dynamic
  EmbeddingGemma-family profile with the built-in profile's thresholds and encode
  entry points.
- Any model name containing `gte-modernbert` resolves to a dynamic
  gte-modernbert-family profile with the built-in profile's calibrated thresholds.
- Other unknown model IDs resolve to the generic profile.

### Local model directories and offline use

Both `check` and `search` accept a directory written by `save_pretrained()` or a
complete Hugging Face repository download. Local paths are passed to Sentence
Transformers with `local_files_only=True`.

```bash
hf download Alibaba-NLP/gte-modernbert-base \
  --local-dir ./models/gte-modernbert-base

codedupes check ./src \
  --model ./models/gte-modernbert-base \
  --device mps

codedupes search ./src "parse json payload" \
  --model ./models/gte-modernbert-base \
  --device mps
```

- Download the complete repository rather than selecting only configuration or
  tokenizer files. Local directories without `config.json` and model weights
  fail before model loading with a corrective error.
- Without `--local-dir`, `hf download <repo-id>` prints the cached snapshot
  directory. That directory can be passed directly to `--model`, including when
  its basename is a commit hash.
- `--model-revision` is ignored for local directories (with a warning): on-disk
  weights have no hub revision. The embedding cache instead keys local models by a
  content fingerprint of the directory, so replacing or retraining the weights in
  place invalidates cached vectors automatically.
- For a Hub model ID rather than a directory path, set
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

For examples, see the [usage guide](usage.md) and [Python API](python-api.md).
