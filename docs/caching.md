# Embedding Cache

`codedupes` persists embedding vectors under `~/.cache/codedupes` so unchanged code never re-embeds across runs. Edit one function and only that function re-embeds next time; when every code unit (and, for `search`, the query) is already cached, the run never loads the model at all.

## Controls

Environment variables:

- `CODEDUPES_CACHE_DIR`: explicit cache root. Takes precedence over everything else.
- `XDG_CACHE_HOME`: when `CODEDUPES_CACHE_DIR` is unset, the cache root is `$XDG_CACHE_HOME/codedupes`.
- Default (both unset): `~/.cache/codedupes`.
- `CODEDUPES_CACHE_MAX_MB`: size cap in megabytes (default `2048`). When a write pushes the cache over the cap, least-recently-used shards are deleted until usage falls to about 80% of the cap. The shard just written is never deleted; if it alone exceeds the cap, it stays usable and a warning recommends raising the limit. A shard that cannot be deleted remains included in the measured total and also produces a warning.
- `CODEDUPES_NO_CACHE=1`: disable the cache for the whole process, every command. Nothing is read or written.

CLI:

```bash
codedupes cache info                               # path, entry count, size, per-model/per-repo breakdown
codedupes cache clear                              # delete every shard for every repo
codedupes cache clear --model gte-modernbert-base  # delete one model's shards
codedupes check ./src --no-cache                   # bypass for one run; on-disk state untouched
```

`codedupes info` also prints a one-line cache summary. Built-in model aliases match case-insensitively in `cache clear --model`; any other model must be passed as the exact string used when analyzing. `--no-cache` works on both `check` and `search`.

## What invalidates what

- Editing a code unit invalidates only that unit's entry; every other unit still hits.
- Different models and different revisions never share entries.
- `check` and `search` embed the corpus under different task instructions, which produce genuinely different vectors, so each command warms its own entries: a warm `check` does not make the first `search` against that corpus warm, and vice versa.
- Repeating an identical search is a full cache hit end to end - query embeddings are cached in the same shard as the corpus they were searched against. The flip side of that shared, immutable shard design: persisting one new query vector rewrites the whole shard matrix on disk, so scripted loops issuing many novel queries against a large corpus pay a full-matrix write per query (repeat queries cost nothing).
- Local model directories (`--model /path/to/model`) are keyed by a content fingerprint of the directory instead of a hub revision, so swapping updated weights into the same path invalidates cached vectors automatically - even when the replacement preserves file sizes and modification times. Metadata-only changes (a touched mtime) do not invalidate anything.
- If you suspect stale results anyway (for example after hand-editing cache files), run `codedupes cache clear`, or add `--no-cache` for a one-off run that bypasses the cache.

## Design notes

Internals for debugging and the curious; none of this is needed to use the cache.

- Entries are content-addressed: each vector is keyed by a hash of (canonical model name, resolved model revision, embedding mode, raw pre-truncation input text) plus a variant fingerprint covering everything else that determines vector values - the encode plan (route and effective prompt), the inference dtype when it differs from the default, the embedding pipeline schema, the installed sentence-transformers/transformers/tokenizers/torch versions, and the remote-code trust setting. Upgrading any of those is a full-corpus miss, never a partial one, so one matrix can never mix vectors from two coordinate systems. EmbeddingGemma uses float32 on CPU and MPS, so those devices share one key space; CUDA bfloat16 vectors stay separate. Keys derive without loading the model, which is what makes the warm no-model-load path possible.
- Vectors live in one shard directory per (analyzed repo root, model, revision):

  ```text
  <cache_root>/repos/<repo-basename>-<pathhash>/<model-slug>@<revision>/
      vectors-<generation>.npy  # immutable float32 matrix
      index.json                # active generation, key -> row map, metadata
  ```

  The path hash keeps two repos that share a directory basename (say, two checkouts both named `src`) from ever colliding.
- Writers publish a complete new vector matrix before atomically switching the index to its generation, and serialize through per-shard advisory file locks, so a concurrent reader can never pair an old key map with rebuilt rows.
- No run deletes rows just because its own invocation didn't include them: one `codedupes` run only ever sees the units selected by its path and filters, never the whole live corpus, so narrowing a run (a subdirectory, an exclude pattern, a single file) leaves sibling vectors warm for the next full run. Rows orphaned by edits or deletions persist until whole-shard LRU eviction against the global byte cap reclaims them; cached search queries are FIFO-capped at 512 per namespace.
- Local model directories are fingerprinted by file contents, not stat metadata. Per-file digests are reused from a manifest under `<cache_root>/local-models/` keyed on the full stat identity (size, mtime, ctime, inode), so unchanged files are never rehashed and warm-path key derivation stays a stat walk. A `--no-cache` run keeps this fingerprint verification in memory without reading or writing the manifest, leaving on-disk cache state untouched. Every local-model load is bracketed by pre- and post-load fingerprints: a directory swapped mid-load triggers one reload from the current on-disk state (a second mid-load change fails the run with a clear error), and cache keys use the fingerprint verified for the weights actually loaded - hits taken under a stale pre-load fingerprint are discarded and everything re-keys, so one result matrix never mixes vectors from two weight sets.
- The builtin profiles pin full commit hashes, which keeps the warm no-model-load path even after the hub cache is cleared. Unpinned revisions (custom models without an explicit `--model-revision`) resolve through the local Hugging Face cache before any model-free hit. After any model load, the true loaded commit hash is double-checked; on a mismatch, pre-load hits are discarded and re-keyed under the true revision so one result matrix never mixes two model revisions. If no concrete commit can be determined at all, persistent reuse is skipped for that call rather than caching under a mutable symbolic name.
- A warm hit for `cpu` or `auto` does not import PyTorch. An explicit `--device mps` or `--device cuda` request validates accelerator availability even on a fully warm cache - an unavailable accelerator errors rather than silently serving cached vectors - which imports PyTorch; resolving a CUDA-specific dtype may also initialize capability checks.
- The cache is never fatal: missing, corrupt, or unwritable cache state degrades to a cache miss (or a skipped write), warns once per process, and analysis proceeds with correct results either way. Every row's digest is recorded at write time and re-verified on read, so a poisoned (NaN/Inf) row *or* a finite row whose bytes changed on disk reads as a per-key miss and is overwritten in place by the next successful recompute - corruption self-heals instead of forcing a manual `cache clear`.
- Fresh model output is validated centrally before it is cached, compared, or returned: wrong shape or row count fails immediately; NaN/Inf or zero vectors on an accelerator clear the allocator cache and retry the batch once on CPU, and fail rather than cache if CPU output is also invalid. All rows are unit-renormalized so dot-product similarity is always cosine similarity.
