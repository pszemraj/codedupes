# Semantic accelerators and Apple Silicon

Install the supported runtime and verify MPS availability as described in [Installation](install.md). The [CLI reference](cli.md) lists device options, and [model profiles](model-profiles.md) lists model-specific thresholds and tasks.

## Device selection

Both `check` and `search` accept:

```bash
codedupes check ./src --device auto
codedupes check ./src --device mps
codedupes search ./src "normalize request payload" --device mps
```

`auto` is the default and resolves in this order:

1. CUDA when `torch.cuda.is_available()` is true
2. MPS when the MPS backend is available
3. CPU

An explicit unavailable accelerator is an error. `codedupes` does not silently reinterpret `--device mps` as CPU, and the check applies even when a warm embedding cache makes inference unnecessary or extraction finds no semantic units to embed - including a `check` whose empty extraction returns before any embedding work is scheduled, where only combined mode with `--allow-semantic-fallback` downgrades the error to a warning. The only automatic CPU transitions are the documented unsupported-op and out-of-memory recovery paths below.

## Unsupported MPS operators

PyTorch controls unsupported-operator fallback through `PYTORCH_ENABLE_MPS_FALLBACK`.

- With `--device mps`, or `--device auto` on macOS, `codedupes` sets the variable to `1` before importing PyTorch when the variable is otherwise unset.
- `--mps-fallback` explicitly sets it to `1`.
- `--no-mps-fallback` explicitly sets it to `0`.
- With neither flag, an existing environment value is respected.

Set this before any other code imports PyTorch. If a long-lived Python process has already imported PyTorch, changing the setting may require a process restart; `codedupes` emits a warning in that case. The Python API intentionally leaves `PYTORCH_ENABLE_MPS_FALLBACK` at the configured value because PyTorch reads it as process-wide runtime state; the previous environment value is not restored after analysis. Library embedders should configure the policy once during process startup or isolate analyses that need different policies in separate processes.

Unsupported-op fallback is different from out-of-memory recovery. Disabling unsupported-op fallback does not disable the explicit OOM recovery policy described next.

## MPS memory policy and OOM recovery

No allocator cap is imposed by default. On memory-constrained systems, start with a cap of `0.9`:

```bash
codedupes check ./src --device mps --mps-memory-fraction 0.9
```

The option calls `torch.mps.set_per_process_memory_fraction()` and accepts `(0, 2]`. `codedupes` rejects `0` because PyTorch defines it as unlimited allocation, which can permit a system-wide OOM. Values above `1` are accepted for parity with PyTorch but emit a warning because they exceed the device-recommended working-set size. A cap can cause an earlier, controlled OOM; it is not a performance setting. The setting is process-global: after codedupes applies a custom cap, the next run whose configuration leaves the option unset restores the allocator baseline captured from `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, or PyTorch's `1.7` default when the environment is unset - including fully cache-covered runs and warm query hits, which never prepare a device. `clear_model_cache()` releases weights but does not itself change allocator policy.

Inference OOM recovery is deterministic. An MPS `Invalid buffer size` failure - a single tensor above Metal's per-buffer cap, raised without any "out of memory" phrase - classifies as MPS OOM and recovers through the same ladder:

1. Detach the failed traceback so temporary tensors are no longer retained by Python frames.
2. Log one warning per failed attempt, including MPS tensor, driver, and recommended-memory statistics when available.
3. Synchronize queued MPS work, run garbage collection, and call `torch.mps.empty_cache()`.
4. Halve the embedding batch size until it reaches one.
5. If an accelerator still OOMs at batch size one, move the cached model to CPU once and retry from the originally requested batch size capped at 32 (`CPU_FALLBACK_MAX_BATCH_SIZE`); host memory has different limits, but host OOM can arrive as an uncatchable OOM-killer kill rather than a Python exception, so an accelerator-sized request (say 512) never carries over. A catchable CPU OOM re-enters the halving ladder above before aborting. The move re-checks the CPU bfloat16 inference policy described below: a model loaded in bfloat16 is cast to float32 unless the experimental opt-in is set and this CPU passes the capability gate.

A model-loading MPS OOM has no batch to shrink, so it clears the MPS cache and retries loading once on CPU. After an MPS-to-CPU OOM fallback, the CPU model remains sticky for that model in a long-lived process. Call `codedupes.semantic.clear_model_cache()` to force a fresh accelerator load.

Successful batches do not clear the allocator cache. Embeddings are converted to normalized NumPy arrays immediately, so pairwise similarity runs on CPU and no large embedding tensor remains resident in Metal memory.

Fresh embeddings must have the expected shape and row count. Non-finite or zero accelerator output retries once on CPU using the same capped batch policy; invalid CPU output fails. Valid rows become unit-normalized float32 arrays, making dot products cosine similarities. Cache-row repair is described under [storage and consistency](caching.md#storage-and-consistency).

## Precision and Metal environment variables

Model loads pin an explicit dtype instead of inheriting the checkpoint's config-declared one, under a capability-gated policy rather than a hardcoded per-device truth.

Without the pin, Transformers 5's `dtype="auto"` default runs float16-configured checkpoints (including the default `gte-modernbert-base`) in half precision - about 10x slower on CPU and off the faithful-float32 tolerance.

CUDA hardware with native bf16 support (Ampere or newer; pre-Ampere emulated bf16 is excluded) always pins bfloat16 - this rule is unchanged.

MPS always pins float32: bfloat16 on MPS gains only ~13% runtime while drifting pair similarities ~1e-2 (tuned-threshold scale), not worth cold-splitting the shared CPU/MPS cache key space.

CPU pins float32 by default. Setting `CODEDUPES_CPU_BF16=1` (experimental) enables bfloat16 when this machine also passes a two-part capability gate - a native bf16 ISA (`bf16` on ARM, `amx_bf16`/`avx512_bf16` on x86) and a GEMM backend able to exploit it (`torch.backends.mkldnn.is_available()`). `torch.backends.cpu.get_cpu_capability()` is never used for this decision: it reports the wheel's build-tier baseline (for example `"DEFAULT"`), not what the running CPU can execute.

The opt-in guard exists because the positive path is unvalidated: the gate proves the CPU can execute bf16 GEMM fast, not that the float32-calibrated duplicate and search thresholds survive bfloat16's numeric shift on the built-in models. Automatic enablement waits for a gate-passing machine (AMX/AVX512-bf16 x86, or ARM bf16 with an mkldnn backend) to validate speed and decision parity end to end. TODO for that promotion work: opted-in CPU bfloat16 currently shares the bfloat16 cache key space with CUDA bfloat16 even though the two backends route ops differently; measure cross-backend row agreement there and split the identities if it matters.

Measured on an Apple M5 (torch 2.13.0, macOS arm64 wheel): `torch.cpu.get_capabilities()` reports a native bf16 ISA (`bf16: true`, `architecture: "arm64"`) but no mkldnn backend, and a 1024x1024x1024 bf16 matmul measured 1015 ms versus 1.207 ms for float32 - 841x slower with an ISA but no backend to exploit it, so this machine's CPU pins float32 with or without the opt-in.

`codedupes info` prints the live verdict: CPU name and architecture, whether the native bf16 ISA is present, whether mkldnn is available, the combined gate, and the effective inference policy (opt-in plus gate).

The gate probes live, at most once per process, and persists nothing: an opted-in run reads `torch.cpu.get_capabilities()` and `torch.backends.mkldnn.is_available()` directly on first use and memoizes the verdict for the rest of the process, so a replaced torch wheel, a changed container CPU mask, or a migrated environment can never serve a stale verdict from disk. A run without the opt-in never imports torch for this decision at all. Only opted-in users pay the per-process probe - proportional for an experimental flag, and simpler than a persisted record whose identity would have to fingerprint the actual wheel and exposed CPU features to be trustworthy.

Every accelerator-to-CPU OOM fallback (see the recovery ladder above) re-checks this same inference policy (opt-in plus gate) before deciding whether to keep or cast away bfloat16, and a load-time OOM retry (CUDA or MPS falling back to a CPU load) re-pins the CPU dtype fresh rather than inheriting the accelerator's dtype.

A run keyed under a non-default (bfloat16) dtype variant whose live execution can no longer produce bfloat16 - an accelerator OOM cast the model to float32 mid-run - discards any cache hits recorded under that key and recomputes the whole corpus in one coherent policy, mirroring the fast-math precedent below; a write that would otherwise land in the wrong key space is skipped instead, costing a cache miss next run rather than a poisoned key space.

The restarted corpus records its faithful CPU identity and stays directly searchable: queries follow that recorded policy even while the analyzer still requests the accelerator. Conversely, a query whose own encode falls back and casts to float32 against a corpus still keyed bfloat16 aborts before the similarity comparison - the correctness boundary is the dot product, not just the cache key.

`codedupes` deliberately does not set `PYTORCH_MPS_FAST_MATH` or `PYTORCH_MPS_PREFER_METAL`. Fast math may change floating-point results around tuned similarity thresholds, while forcing a particular matmul implementation is a workload-specific optimization. You can experiment with those variables externally, but re-run the hybrid tuning guardrail and a representative repository before adopting altered thresholds. The persistent embedding cache keys `PYTORCH_MPS_FAST_MATH` into its vector identity whenever the request could execute on MPS (explicit `mps`, or `auto` on macOS), so toggling the policy re-embeds instead of serving vectors from the other math mode; the key mirrors torch's exact rule, where any set value except the literal `0` enables fast math (an empty string enables it). If a fast-math corpus run executes off MPS (because `auto` resolves elsewhere or inference falls back after an OOM), codedupes discards any fast-math hits/results and rebuilds the complete matrix under the faithful CPU cache identity. Its queries stay on that recorded CPU policy. A standalone fast-math query that leaves MPS aborts before the dot product because the caller's matrix policy cannot be proven. `PYTORCH_MPS_PREFER_METAL` (presence-only: setting it to `0` still enables it) only selects among faithful float32 implementations and intentionally shares the key space, like CPU and MPS float32 do.

For a native macOS installation, use the default `gte-modernbert-base` profile first; evaluate `embeddinggemma-300m` only after the default path is stable.

## MLX coexistence

MLX and PyTorch both consume Apple unified memory but manage it separately. `codedupes` never imports MLX and never touches its allocator; if MLX is already loaded in the process and semantic execution resolves to MPS, `codedupes` logs one warning about shared unified-memory pressure. Releasing MLX arrays and clearing MLX caches remains the host application's job.

## Hardware validation

MPS behavior is validated on real Apple Silicon hardware only - the test suite contains no simulated MPS. `tests/test_semantic_mps.py` runs automatically wherever PyTorch reports a usable MPS device and skips only where the hardware is genuinely absent (a non-Mac host, or a sandbox that blocks Metal device access - a skipped run performs zero MPS validation, so run it from an environment with device access):

```bash
pytest tests/test_semantic_mps.py
```

The suite loads the pinned default model on `mps`, checks CPU/MPS embedding parity, validates explicit `--device mps` requests against a warm embedding cache, and provokes genuine Metal allocator out-of-memory (by lowering `torch.mps.set_per_process_memory_fraction`) to prove load-time CPU fallback, the batch-halving ladder, and query-encode recovery work on the real allocator. The default model must already be cached locally (any prior `codedupes check` or `hf download` does this).

CUDA behavior is validated the same way, on real GPUs only. `tests/test_semantic_cuda.py` runs automatically wherever `torch.cuda.is_available()` is true and skips otherwise:

```bash
pytest tests/test_semantic_cuda.py
```

It covers the five behaviors this documentation advertises for CUDA hosts: bfloat16 is pinned only where `torch.cuda.is_bf16_supported(including_emulation=False)` is true and the loaded model's live parameter dtype agrees; cold CUDA inference completes and tracks CPU within the tolerance its dtype allows; a genuine allocator out-of-memory (provoked by lowering `torch.cuda.set_per_process_memory_fraction`) drives the load-time CPU fallback with a re-pinned CPU dtype, the batch-halving ladder, and the capped CPU restart; a keyed-bfloat16 corpus that lands on CPU rebuilds as one coherent float32 matrix that stays searchable; and a query whose fallback casts it to float32 aborts before the dot product instead of comparing across policies. Tests that require native bfloat16 skip on pre-Ampere hardware, where CUDA keys as float32 and those paths do not exist. The bfloat16 tests are the only CUDA-specific coverage of the dtype re-pin: MPS always resolves float32, so the MPS suite cannot exercise it.

A companion opt-in smoke test validates every built-in profile against the multi-domain probe corpus in `test_fixtures/search_probes/`: every relevant query must surface its expected function at that profile's default search threshold and every off-topic query must return nothing:

```bash
CODEDUPES_SMOKE_SEARCH=1 pytest tests/test_semantic_smoke.py
```

## Upstream references

- [PyTorch 2.13 release notes](https://pytorch.org/blog/pytorch-2-13-release-blog/)
- [PyTorch 2.13 MPS backend requirements](https://docs.pytorch.org/docs/2.13/notes/mps.html)
- [PyTorch 2.13 MPS environment variables](https://docs.pytorch.org/docs/2.13/mps_environment_variables.html)
- [PyTorch 2.13 `torch.mps` API](https://docs.pytorch.org/docs/2.13/mps.html)
- [SentenceTransformer device placement](https://sbert.net/docs/package_reference/sentence_transformer/model.html)
- [MLX Metal memory APIs](https://ml-explore.github.io/mlx/build/html/python/metal.html)
- [MLX compiled-function caching](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
