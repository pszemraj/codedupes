# Semantic model profiles and tasks

Profiles resolve model aliases, thresholds, revisions, trust settings, and task-specific embedding behavior. See [Installation](install.md) for dependencies, the [CLI reference](cli.md) for option syntax, and [Accelerators](accelerators.md) for device and precision behavior.

Most users should leave model and task settings unset. `codedupes` uses the pinned `gte-modernbert-base` profile and downloads it automatically on the first semantic run. Run `codedupes info` to see the effective profile, checkpoint, device, and installed runtime. Choose another model only when you have a reason to evaluate its results or need an already-downloaded local copy.

## Built-in profiles

| profile key | canonical model ID | family | search threshold | default revision | default trust mode |
| --- | --- | --- | --- | --- | --- |
| `gte-modernbert-base` | `Alibaba-NLP/gte-modernbert-base` | `gte-modernbert` | `0.50` | `e7f32e3c00f91d699e8c43b53106206bcc72bb22` | `False` |
| `embeddinggemma-300m` | `unsloth/embeddinggemma-300m` | `embeddinggemma` | `0.40` | `bfa3c846ac738e62aa61806ef9112d34acb1dc5a` | `False` |

- [Per-language duplicate gates and their selection policy](analysis-defaults.md#semantic-duplicate-gate-defaults) control `check` reporting. The table's search threshold is only the floor for query matches; query-to-code similarity is much lower than code-to-code duplicate similarity.
- Every builtin default revision is a pinned immutable commit. [Calibration sweeps](hybrid-tuning.md#semantic-threshold-sweep-model-profiles) record the checkpoint, prompt plan, pipeline, and candidate policy behind each threshold.
- Search defaults favor recall. Inspect scores on representative queries and raise `--semantic-threshold` (or the Python API's per-query `threshold`) if results are too broad; no fixed floor separates relevant and off-topic code on every repository. The multi-domain probes in `test_fixtures/search_probes/` check the built-in search floors; the single-domain [calibration sweeps](hybrid-tuning.md#semantic-threshold-sweep-model-profiles) are additional guardrails, not the source of those floors.
- Generic/unknown models fall back to duplicate threshold `0.82` and search threshold `0.35` unless you override `--semantic-threshold` / `semantic_threshold`.

## Alias resolution rules

- Built-in alias keys and known aliases resolve to the profile's canonical model ID.
- Built-in aliases and Hub IDs cannot be shadowed by same-named directories in the current working directory. A local model must use explicit path syntax (an absolute path, `./` or `../`, or `~`); it is then canonicalized to its resolved absolute path - including the on-disk letter case on case-insensitive filesystems such as macOS - so equivalent explicit spellings share one cache identity.
- Known local model families are inferred from a recognizable directory name, Hugging Face cache ancestor, saved configuration, or model-card title.
- Family inference selects loading and prompt behavior only. Non-builtin models, including fine-tunes and local copies, use the [generic thresholds](#built-in-profiles). Calibrated thresholds belong to the pinned builtin checkpoint; a recognizable name does not establish calibration. A family-matched model warns once per model per process about its generic duplicate gate. Pass `--threshold`/`--semantic-threshold` for tuned weights.

### Local model directories and offline use

Use a Hub model ID for the normal online path; manual downloads are only needed for offline use or a custom checkpoint. Both `check` and `search` accept a directory written by `save_pretrained()` or a complete Hugging Face repository download, using the [explicit path rules](#alias-resolution-rules). Local paths are passed to Sentence Transformers with `local_files_only=True`.

```bash
hf download Alibaba-NLP/gte-modernbert-base \
  --local-dir ./models/gte-modernbert-base

codedupes check ./src \
  --model ./models/gte-modernbert-base

codedupes search ./src "parse json payload" \
  --model ./models/gte-modernbert-base
```

- Download the complete repository rather than selecting only configuration or tokenizer files. Local directories without `config.json` and model weights fail before model loading with a corrective error.
- Without `--local-dir`, `hf download <repo-id>` prints the cached snapshot directory. That directory can be passed directly to `--model`, including when its basename is a commit hash.
- `--model-revision` is ignored for local directories (with a warning): on-disk weights have no hub revision. See [local-model cache invalidation](caching.md#what-invalidates-what) for how weight changes affect cached vectors.
- For a Hub model ID rather than a directory path, set `HF_HUB_OFFLINE=1` to guarantee no network access.

For live effective values in your environment, run:

```bash
codedupes info
```

## Semantic task defaults and choices

Leave `semantic_task` unset unless you are deliberately changing a model's embedding behavior. `analyze()`/`codedupes check` use `semantic-similarity`; `index()`/`codedupes search` use `code-retrieval`. Those defaults align the model's prompt and encode route with the operation.

CLI task defaults:

- `codedupes check`: `semantic-similarity`
- `codedupes search`: `code-retrieval`

The Python API resolves the same defaults by operation: an unset `AnalyzerConfig.semantic_task` uses `semantic-similarity` for `CodeAnalyzer.analyze()` and `code-retrieval` for `CodeAnalyzer.index()`. A later `search()` uses the task that produced its current corpus embeddings. Built-in default thresholds apply only to the pinned revision, default task/prompt plan, and default remote-code setting they were calibrated on; a custom instruction prefix, alternate EmbeddingGemma task, alternate built-in revision, or a `trust_remote_code` value differing from the profile default requires an explicit threshold. Remote code can change the vectors, so it splits the embedding cache key and invalidates the calibrated defaults the same way a prompt change does; this applies to both the duplicate gates and the search default.

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
