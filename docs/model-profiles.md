# Semantic model profiles and tasks

Profiles resolve model aliases, thresholds, revisions, trust settings, and task-specific embedding behavior. See [Installation](install.md) for dependencies, the [CLI reference](cli.md) for option syntax, and [Accelerators](accelerators.md) for device and precision behavior.

Most users should leave model and task settings unset. `codedupes` uses the pinned `gte-modernbert-base` profile and downloads it automatically on the first semantic run. Run `codedupes info --verbose` to see the effective profile, checkpoint, device, and installed runtime. Choose another model only when you have a reason to evaluate its results or need an already-downloaded local copy.

## Built-in profiles

| profile key | canonical model ID | family | search threshold | default revision | default trust mode |
| --- | --- | --- | --- | --- | --- |
| `gte-modernbert-base` | `Alibaba-NLP/gte-modernbert-base` | `gte-modernbert` | `0.50` | `e7f32e3c00f91d699e8c43b53106206bcc72bb22` | `False` |
| `embeddinggemma-300m` | `unsloth/embeddinggemma-300m` | `embeddinggemma` | `0.40` | `bfa3c846ac738e62aa61806ef9112d34acb1dc5a` | `False` |

- [Per-language duplicate gates and their selection policy](analysis-defaults.md#semantic-duplicate-gate-defaults) control `check` reporting. The table's search threshold is only the floor for query matches; query-to-code similarity is much lower than code-to-code duplicate similarity.
- Every built-in default revision is a pinned immutable commit. [Calibration sweeps](hybrid-tuning.md#semantic-threshold-sweep-model-profiles) record the checkpoint, prompt plan, pipeline, and candidate policy behind each threshold.
- Search defaults favor recall. Inspect scores on representative queries and raise `--semantic-threshold` (or the Python API's per-query `threshold`) if results are too broad; no fixed floor separates relevant and off-topic code on every repository. The multi-domain probes in `test_fixtures/search_probes/` check the built-in search floors; the single-domain [calibration sweeps](hybrid-tuning.md#semantic-threshold-sweep-model-profiles) are additional guardrails, not the source of those floors.

## Alias resolution rules

- Built-in alias keys and known aliases resolve to the profile's canonical model ID.
- Built-in aliases and Hub IDs cannot be shadowed by same-named directories in the current working directory. A local model must use explicit path syntax (an absolute path, `./` or `../`, or `~`); it is then canonicalized to its resolved absolute path - including the on-disk letter case on case-insensitive filesystems such as macOS - so equivalent explicit spellings share one cache identity.
- Known local model families are inferred from saved configuration first, then a recognizable directory name, Hugging Face cache ancestor, or top-level `# ` model-card heading within the first 128 lines. EmbeddingGemma is recognized from its `gemma3_text` configuration with bidirectional attention even in an arbitrarily named directory without a README. A plain ModernBERT architecture does not establish GTE identity.
- Recognized copies and fine-tunes use their family's loading/prompt behavior and tuned thresholds by default. Recognition works offline and does not verify exact checkpoint equivalence: family tuning is a practical starting point, not a guarantee of identical score distributions. Actual local paths and non-builtin Hub IDs are preserved without inheriting the built-in revision pin.
- Automatically selecting family defaults for a non-builtin Hub model emits a warning once per model per process about possible score-distribution differences. Built-in defaults need no generic-policy hint; recognized local copies get that hint once when INFO logging is enabled. Explicit numeric or named/generic profile choices do not emit these notices.

### Choosing threshold defaults

Both `check` and `search` accept `--threshold-profile`; the Python setting is `threshold_profile` (default `"auto"`).

| choice | threshold behavior |
| --- | --- |
| `auto` | Use the recognized model family's thresholds, or generic defaults for an unknown model. |
| `generic` | Use duplicate gate `0.82` for every language and search floor `0.35`. |
| `embeddinggemma-300m` | Use the built-in EmbeddingGemma profile's thresholds. |
| `gte-modernbert-base` | Use the built-in GTE profile's thresholds. |

Explicit numeric thresholds take precedence. Selecting a threshold profile changes only result filtering; it does not change the model, prompts, revision, or cached embeddings. It also does not bypass the explicit numeric threshold requirements for [custom embedding contexts](#semantic-task-defaults-and-choices). In human-readable output, the CLI reports its effective threshold choice and values without prompting; `--json` suppresses those logs and does not add threshold metadata to the JSON schema.

For an approved local EmbeddingGemma copy, normal family recognition is enough:

```bash
codedupes check ./src --model ./models/approved-copy
codedupes check ./src --model ./models/approved-copy --threshold-profile generic
```

### Local model directories and offline use

Use a Hub model ID for the normal online path; Hub access downloads model assets, while source and queries are embedded locally. Manual downloads are only needed for offline use or a custom checkpoint. Both `check` and `search` accept a directory written by `save_pretrained()` or a complete Hugging Face repository download, using the [explicit path rules](#alias-resolution-rules). Local paths are passed to Sentence Transformers with `local_files_only=True`.

```bash
hf download Alibaba-NLP/gte-modernbert-base \
  --local-dir ./models/gte-modernbert-base

codedupes check ./src \
  --model ./models/gte-modernbert-base
```

- Download the complete repository rather than selecting only configuration or tokenizer files. Local directories without `config.json` and model weights fail before model loading with a corrective error.
- Without `--local-dir`, `hf download <repo-id>` prints the cached snapshot directory. That directory can be passed directly to `--model`, including when its basename is a commit hash.
- `--model-revision` is ignored for local directories (with a warning): on-disk weights have no hub revision. See [local-model cache invalidation](caching.md#what-invalidates-what) for how weight changes affect cached vectors.
- For a Hub model ID rather than a directory path, set `HF_HUB_OFFLINE=1` to guarantee no network access.

## Semantic task defaults and choices

Leave `semantic_task` unset unless you are deliberately changing a model's embedding behavior. `analyze()`/`codedupes check` use `semantic-similarity`; `index()`/`codedupes search` use `code-retrieval`; a later `search()` uses the task that produced its current corpus embeddings. Those defaults align the model's prompt and encode route with the operation.

The shipped thresholds were calibrated on the pinned built-in checkpoints and are also offered as family defaults for recognized copies. A custom instruction prefix, alternate EmbeddingGemma task, alternate built-in revision, or a `trust_remote_code` value differing from the model profile default still requires an explicit numeric threshold, regardless of `threshold_profile`. Remote code can change the vectors, so it splits the embedding cache key and invalidates the threshold defaults the same way a prompt change does; this applies to both the duplicate gates and the search default.

[Contextual search documents](python-api.md#semantic-query-search) also require an explicit threshold because the built-in search defaults were calibrated on source-only documents.

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
- GTE ModernBERT and generic models: symmetric `encode` route with no prompt unless explicitly overridden.
- `--instruction-prefix` replaces the model prompt for that input mode while preserving the encode route; it is never stacked inside the saved prompt.

See [cache identity](caching.md#runtime-identity) for how encode routes and prompts affect reuse.

For invocation examples and direct-call embedding identity requirements, see the [CLI reference](cli.md) and [Python search API](python-api.md#semantic-query-search).
