# Embedding Cache

`codedupes` persists computed embedding vectors to disk so unchanged code units do not re-embed
across runs.

## How it works

- The cache is **content-addressed**: each cached vector is keyed by a hash of
  `(canonical model name, resolved model revision, embedding mode, prepared embedding
  text)`. The prepared text is the pre-truncation output of
  `prepare_code_for_embedding(...)`, so the key can be derived without loading the
  model. For model families whose torch dtype depends on the execution device
  (currently EmbeddingGemma), the resolved device and selected dtype are also part
  of the key. An `auto` request therefore resolves to CPU, CUDA, or MPS before
  lookup, and vectors computed under bfloat16 are never served for a float32 run;
  `gte-modernbert-base` embeds identically across devices and shares one key space.
- Because the key depends on the exact prepared text, **partial updates happen
  naturally**: editing one function in one file only invalidates that function's cache
  entry; every other unit in the corpus still hits.
- Cached vectors live under one directory ("shard") per `(analyzed repo root, model,
  revision)` combination:

  ```text
  <cache_root>/repos/<repo-basename>-<pathhash>/<model-slug>@<revision>/
      vectors.npy   # float32 matrix, one row per cached embedding
      index.json    # key -> row map, plus schema/model/revision/last_used_at
  ```

  The path hash means two repos that happen to share a directory basename (for
  example two checkouts both named `src`) never collide, and identical code cached
  under the same model/revision is naturally shared within a repo's shard.
- **No model load on a full cache hit.** `codedupes check`/`search` resolve the
  model revision, prepare embedding text, and check the cache *before* touching
  `sentence-transformers`. If every code unit (and, for `search`, the query) is
  already cached, the model is never loaded at all. A warm `check`/`search` run
  skips model load and inference entirely. Device-sensitive families may initialize
  PyTorch capability checks to resolve `auto` and select the cache variant, but do
  not load model weights.
- Query embeddings from `codedupes search` are cached in the same shard as the
  corpus they were searched against, keyed on the prepared query text. Repeating an
  identical search is a full cache hit end-to-end.
- `check` and `search` embed the corpus under different task instructions
  (`semantic-similarity` vs `code-retrieval`), which produce genuinely different
  vectors, so each command warms its own entries: a warm `check` does not make the
  first `search` against that corpus warm, and vice versa.
- Local model directories (`--model /path/to/model`) are keyed by a content
  fingerprint of the directory (file names, sizes, and mtimes) instead of a hub
  revision, so swapping updated weights into the same path invalidates the cache
  automatically while unchanged directories keep the skip-model-load fast path.
- If the model resolves to an unpinned revision (for example the default
  `gte-modernbert-base` profile), the cache resolves the locally cached HuggingFace
  commit hash from disk before loading the model. If that can't be determined
  offline, the run falls back to loading the model normally. After any model load,
  the cache double-checks the true loaded commit hash against what it assumed; on a
  confirmed mismatch it discards any pre-load cache hits and re-keys under the true
  revision (which may itself hit entries cached earlier under that revision), so a
  single result matrix is never assembled from two different model revisions. Note
  that pinning `--revision` to a moving branch name like `main` keys the cache by
  that literal string, which disables the skip-model-load fast path (a commit-hash
  pin or the unpinned default keeps it).
- The cache is **never fatal**. A missing, corrupt, or unwritable cache file is
  treated as a cache miss (or a no-op write) and logged once per process as a
  warning; analysis always proceeds and produces correct results either way.

## Location and environment variables

- `CODEDUPES_CACHE_DIR`: explicit cache root directory. Takes precedence over
  everything else.
- `XDG_CACHE_HOME`: when `CODEDUPES_CACHE_DIR` is unset, the cache root is
  `$XDG_CACHE_HOME/codedupes`.
- Default (both unset): `~/.cache/codedupes`.
- `CODEDUPES_CACHE_MAX_MB`: opportunistic size cap in megabytes (default `2048`).
  After a write, if the cache exceeds this cap, the least-recently-used shards are
  deleted until the cache is back under about 80% of the cap. The shard just
  written is preserved; if that shard alone exceeds the cap, it remains usable and
  a warning recommends raising the limit.
- `CODEDUPES_NO_CACHE=1`: disable the embedding cache globally for the process, for
  every command. No cache files are read or written.

## CLI

```bash
codedupes cache info
codedupes cache clear
codedupes cache clear --model gte-modernbert-base
```

- `codedupes cache info`: prints the cache path, whether it is disabled via
  `CODEDUPES_NO_CACHE`, total entry count, size on disk, and a per-model and
  per-repo breakdown.
- `codedupes cache clear [--model <name>]`: deletes cached embeddings. Without
  `--model`, clears every shard across every analyzed repo. With `--model`, only
  shards for that model (alias or canonical HuggingFace ID) are removed. Prints the
  number of entries cleared. Built-in aliases match case-insensitively; a
  non-builtin model must be passed as the exact string used when analyzing.
- `codedupes info` also prints a short embedding-cache summary line (path, entry
  count, size on disk).

`--no-cache` is available on both `codedupes check` and `codedupes search` to
disable the cache for a single invocation without touching any on-disk state.

## When results look stale

The cache is keyed by exact prepared text and resolved model revision, so it should
never serve embeddings computed from a different model or a different version of the
code. If you ever suspect stale results anyway (for example after manually rewriting
cache files, or after switching HuggingFace cache directories), the fix is:

```bash
codedupes cache clear
```

or `codedupes check ./src --no-cache` / `codedupes search ./src "..." --no-cache` for
a one-off run that bypasses the cache entirely.
