# Semantic Model Profiles and Tasks

Profiles resolve model aliases, thresholds, revisions, trust settings, and task-specific embedding behavior. See [Installation](install.md) for dependencies, the [CLI reference](cli.md) for option syntax, and [Accelerators](accelerators.md) for device and precision behavior.

## Built-in profiles

| profile key | canonical model ID | family | duplicate threshold | search threshold | default revision | default trust mode |
| --- | --- | --- | --- | --- | --- | --- |
| `gte-modernbert-base` | `Alibaba-NLP/gte-modernbert-base` | `gte-modernbert` | `0.96` | `0.50` | `e7f32e3c00f91d699e8c43b53106206bcc72bb22` | `False` |
| `embeddinggemma-300m` | `unsloth/embeddinggemma-300m` | `embeddinggemma` | `0.86` | `0.52` | `bfa3c846ac738e62aa61806ef9112d34acb1dc5a` | `False` |

Notes:

- The duplicate threshold gates `check` pair reporting; the search threshold is the floor for `search` query matches. Query-to-code similarity runs far below code-to-code duplicate similarity, so search defaults are intentionally much lower.
- Every builtin default revision is a pinned immutable commit: a calibrated threshold is only a property of the exact checkpoint, prompt plan, and pipeline it was swept on, and the calibration identity for each default is recorded in `test_fixtures/hybrid_tuning/semantic_threshold_report.json` (duplicates) and `search_threshold_report.json` (search). Pinning also keeps the warm no-model-load cache path stable after the hub cache is cleared.
- The gte-modernbert search default is calibrated recall-safe against real corpora: genuinely relevant hits start near `0.59` while fully off-topic queries ceiling near `0.48` on most corpora. Vocabulary overlap from a shared domain (GPU kernels vs a graphics query) or even a single shared word ("pattern", "parse", "warp") can push off-topic matches to `0.52`-`0.65`; those carry visible scores and rank below real hits, but raise `--semantic-threshold` toward `0.6` if they clutter results. No fixed floor separates them everywhere, and the default deliberately favors recall over precision. The synthetic-corpus search sweep saturates for this model (every unit shares one domain), so its report is a guardrail, not the default's source.
- The EmbeddingGemma search default comes from the labeled probe sweep under the fixed prompt pipeline: `0.54` maximizes F1, and the default takes one step looser (`0.52`, full probe recall) per the recall-first policy. The pre-prompt-fix `0.40` belonged to a different vector space and was not carried forward.
- Generic/unknown models fall back to duplicate threshold `0.82` and search threshold `0.35` unless you override `--semantic-threshold` / `semantic_threshold`.

## Alias resolution rules

- Built-in alias keys and known aliases resolve to the profile's canonical model ID.
- A model name that is an existing directory on disk is treated as a local model copy before built-in aliases are considered, then canonicalized to its resolved absolute path - including the on-disk letter case on case-insensitive filesystems such as macOS - so relative, absolute, and differently-cased spellings share one cache identity.
- Known local model families are inferred from a recognizable directory name, Hugging Face cache ancestor, saved configuration, or model-card title.
- Family inference selects loading and prompt behavior only. Any non-builtin model - a hub name or local directory containing `embeddinggemma` or `gte-modernbert`, a fine-tune, an arbitrary copy - keeps the family's encode entry points and prompts but uses the uncalibrated generic thresholds: calibrated thresholds belong to the exact pinned builtin checkpoint they were swept on, and a lookalike name proves nothing about a model's score distribution. Pass `--threshold`/`--semantic-threshold` for tuned weights.
- Other unknown model IDs resolve to the generic profile.

### Local model directories and offline use

Both `check` and `search` accept a directory written by `save_pretrained()` or a complete Hugging Face repository download. Local paths are passed to Sentence Transformers with `local_files_only=True`.

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

- Download the complete repository rather than selecting only configuration or tokenizer files. Local directories without `config.json` and model weights fail before model loading with a corrective error.
- Without `--local-dir`, `hf download <repo-id>` prints the cached snapshot directory. That directory can be passed directly to `--model`, including when its basename is a commit hash.
- `--model-revision` is ignored for local directories (with a warning): on-disk weights have no hub revision. The embedding cache instead keys local models by a content fingerprint of the directory, so replacing or retraining the weights in place invalidates cached vectors automatically.
- For a Hub model ID rather than a directory path, set `HF_HUB_OFFLINE=1` to guarantee no network access.

For live effective values in your environment, run:

```bash
codedupes info
```

## Semantic task defaults and choices

CLI task defaults:

- `codedupes check`: `semantic-similarity`
- `codedupes search`: `code-retrieval`

The Python API resolves the same defaults by operation: an unset `AnalyzerConfig.semantic_task` uses `semantic-similarity` for `CodeAnalyzer.analyze()` and `code-retrieval` for `CodeAnalyzer.index()`. A later `search()` uses the task that produced its current corpus embeddings; an explicit task overrides either default.

Allowed task names:

- `semantic-similarity`
- `code-retrieval`
- `retrieval`
- `question-answering`
- `fact-verification`
- `classification`
- `clustering`

If you pass an unknown semantic task, the CLI/API raises a validation error.

## Task/prompt behavior by model family

Prompts are backend configuration, not text decoration: codedupes passes raw code/query text to Sentence Transformers together with an explicit prompt and encode route, so the model's saved prompts are never applied a second time on top of a manually prefixed input.

- `embeddinggemma`: duplicate detection embeds code through the symmetric `encode` route with the task prompt (for example `task: sentence similarity | query: ` for `semantic-similarity`); retrieval-task code inputs use `encode_document` with the document prompt (`title: none | text: `); queries use `encode_query` with the task's query prompt (for example `task: code retrieval | query: ` for `code-retrieval`).
- generic models: symmetric `encode` route with no prompt unless explicitly overridden.
- `--instruction-prefix` replaces the model prompt for that input mode while preserving the encode route; it is never stacked inside the saved prompt.

The encode route and effective prompt participate in embedding-cache identity, so changing task, prompt, or route can never reuse vectors produced under a different plan.

For examples, see the [usage guide](usage.md) and [Python API](python-api.md).
