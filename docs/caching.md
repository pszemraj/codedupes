# Embedding Cache

`codedupes` persists embeddings so unchanged code can reuse vectors across runs. A fully cached corpus and query can run without loading the model.

## Controls

| Variable | Behavior |
| --- | --- |
| `CODEDUPES_CACHE_DIR` | Explicit cache root; takes precedence over other settings. |
| `XDG_CACHE_HOME` | Uses `$XDG_CACHE_HOME/codedupes` when no explicit root is set. Otherwise defaults to `~/.cache/codedupes`. |
| `CODEDUPES_CACHE_MAX_MB` | Global size cap, default `2048` MB. Values must be at least `1` and are floored to whole MB. Invalid values warn once per process and use the default. |
| `CODEDUPES_NO_CACHE=1` | Disables persistent cache reads and writes for the process. |

`--no-cache` bypasses persistent storage for one `check` or `search` run without changing existing files. Use the [cache commands](cli.md#codedupes-cache-info) to inspect usage or clear entries. Built-in aliases match case-insensitively in `cache clear --model`; pass other model names exactly as used for analysis.

After each nonempty batch write, codedupes inventories shard sizes. When the global cap is exceeded, it removes least-recently-used shards toward 80% of the cap. The shard just written is protected, even if it alone exceeds the cap; an oversized or undeletable shard produces a warning and remains included in usage. Inventory reads the filesystem so cooperating processes see each other's writes.

## What invalidates what

| Change | Cache effect |
| --- | --- |
| Edit a unit's prepared input | Re-embeds that input; unchanged inputs still hit. |
| Move or rename source-only code | Reuses the content key; the manifest can report a move. |
| Move a contextual search document | Re-embeds because its path is part of the input. |
| Change model, revision, prompt, encode route, or vector-affecting runtime settings | Uses a different embedding identity. |
| Replace local weights in place | Changes the directory content fingerprint. Touching modification times alone does not invalidate vectors. |
| Repeat a search query | Reuses its query vector when both corpus and query identities match. |

EmbeddingGemma uses different corpus prompts for `check` and `search`, so they warm independently. GTE uses the same symmetric corpus route for both: a warm check can cover the first search's corpus, but a new query still embeds. See [prompt behavior](model-profiles.md#taskprompt-behavior-by-model-family).

Query vectors share the corpus shard. Each novel query write rewrites that shard's full immutable matrix; loops issuing many new queries against a large corpus therefore pay a full-matrix write per query. Query rows have an independent FIFO cap and are excluded from corpus orphan collection.

### Hub revisions

Built-in profiles and explicit full-commit `--model-revision` values use immutable revision keys. Unpinned Hub models default to the requested branch/tag label, or `main`, without an offline revision lookup before a warm hit.

A label-keyed shard records the source commit with its vector generation. When a miss loads the model, codedupes compares the loaded commit with both the current shard and the snapshot that supplied earlier hits. Drift invalidates those hits and rebuilds the corpus. Writers reject batches whose commit became stale during encoding. A backend that cannot report its commit recomputes the complete corpus and bypasses both corpus and query caching.

Fully warm label-keyed runs can keep serving a coherent set of older vectors after a branch moves. Clear that model's cache after a known upstream change to refresh it immediately. `--strict-revision-cache` instead resolves the label through the local Hugging Face cache before reuse and uses that concrete commit for model loading. If the label cannot be resolved offline, persistent reuse is disabled. Built-in pins and local directories are unaffected.

An indexed corpus also retains its source commit. A query from a different checkpoint raises a reindex error before similarity comparison, including when another process has replaced the shard since indexing. See the [search state contract](python-api.md#semantic-query-search).

### Local directories

Local model identity hashes file contents. Per-file digests are reused from `<cache_root>/local-models/` when size, mtime, ctime, and inode match, keeping unchanged runs to a stat walk. A no-cache run maintains this information only in memory; enabling caching later can persist it.

Model loading checks fingerprints before and after reading weights. A change during loading triggers one reload; a second change fails the run. Earlier hits are discarded if their fingerprint differs from the loaded weights.

## Corpus lifecycle

The cache tracks three distinct objects:

- A unit UID identifies one occurrence of code.
- A content key identifies prepared embedding input; several units can share it.
- A vector row stores that key's embedding. Compaction can change its row number.

After a successful `analyze()` or `index()`, `manifest.json` records the unit-to-key map and exact file paths. Failed runs leave the previous manifest in place, even if they already wrote some vectors. Complete scans publish empty selections too, so deleting the final unit still updates the baseline.

Comparable scans match new UIDs to departed UIDs one-to-one by content key to infer moves. Unmatched departed UIDs count as deleted even if another unit still shares their vector. A key becomes orphaned only when the current selection has no unit referencing it.

Selections are compared only when candidate settings and the effective runtime variant agree. These settings include statement/type filters, private/stub/exclude policy, languages, task, prefix, and search document mode. Switching filters, operations, or float32/bfloat16 variants keeps separate baselines. The manifest retains up to 16 recent selections, plus older selections with pending orphan records.

### Complete and partial scans

Directory targets use that directory as their cache scope; file targets use their parent. Thus `check ./src` and `check ./src/a.py` share a shard, while `check .` uses another.

A file target or directory scan with explicit excludes publishes an incomplete observation. It merges into the prior selection instead of deleting unseen siblings. A single-file scan replaces only that file's baseline slice, so observed edits or eligibility changes can orphan old keys. It neither advances the complete-scan clock nor refreshes the pin age of unseen units.

### Orphan collection

The shard-wide manifest generation counts complete scans, independently of vector snapshot IDs. A key orphaned at generation `g` remains available for three further complete scans. Reintroducing its content clears the orphan record and reuses the row.

At `g+3`, collection can remove the row if no recently refreshed selection references it. A selection pins its keys only within the three-scan window; stale selections retain deletion baselines but cannot pin confirmed deletions forever. Their other vectors are not inferred to be deleted and remain available until whole-shard eviction.

Collection rechecks the manifest and vector generation under the shard lock, so a concurrent reintroduction cancels stale cleanup. Query rows remain untouched. See [embedding telemetry](output.md#embedding-telemetry) for move, deletion, and retained/collected row counts.

## Storage and consistency

Each analyzed root, model, and revision has a shard:

```text
<cache_root>/repos/<repo-basename>-<pathhash>/<model-slug>@<revision>/
    vectors-<generation>.npy  # immutable float32 matrix
    index.json                # active generation, content key -> row, metadata
    manifest.json             # corpus references and complete-scan ages
```

The root path hash separates checkouts with the same basename. A writer publishes the full matrix before atomically switching `index.json`; key maps, row digests, and source-commit metadata travel with that snapshot. Readers reconfirm the generation so a concurrent rebuild cannot pair an old map with new rows. Deletion during a read becomes a miss.

Per-shard advisory locks live under `<cache_root>/locks/`, outside shard directories. Clear and eviction lock the shard before deletion, then reclaim its lock file if deletion succeeded. Overlapping deletion and recreation can briefly produce distinct lock inodes; generation checks keep this a miss/recompute case rather than a mismatched vector read. The next locked write removes abandoned temporary files and, when the index is readable, unpublished vector generations.

Cache inspection, eviction, and clearing continue past shards that disappear or become unreadable. Clear reports deletion failures to the CLI instead of claiming success. Analysis-time cache failures warn once per process and degrade to misses or skipped writes. Row digests detect non-finite or changed vector bytes; recomputation repairs those rows. A dimension change replaces an incompatible shard with a warning.

### Runtime identity

Keys cover the canonical model, revision, complete prepared input, encode route/prompt, pipeline schema, dtype variant, library versions, and remote-code trust setting. Old preprocessing schemas cannot reuse current vectors. Deriving keys does not require loading weights.

CPU and MPS float32 share keys; bfloat16 and MPS fast math use separate variants. Device kernels can round differently, so clear the cache when measuring a single-device reference. [Accelerator precision and fallback](accelerators.md#precision-and-metal-environment-variables) explains which policies can share vectors and when a corpus must restart.

A warm CPU run, or `auto` on macOS, can avoid importing PyTorch. The experimental CPU bfloat16 opt-in requires a live capability probe. `auto` elsewhere imports PyTorch for device/dtype resolution; explicit accelerator requests validate availability even when no inference is needed.

### Filesystem permissions

The configured root is resolved to an absolute physical path, so equivalent spellings share locks. It may be a symlink, but managed shard, `repos`, `local-models`, and `locks` directories reject pre-existing symlinks. New cache directories use `0700`; files use `0600`. Existing root permissions are retained. Advisory locks coordinate cooperating processes; they do not secure an attacker-writable parent directory.
