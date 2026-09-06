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

An explicit unavailable accelerator is an error, including on warm-cache and empty scans. Combined mode can retain traditional results with `--allow-semantic-fallback`; see [exit codes](output.md#exit-codes). Automatic CPU transitions during inference follow the recovery rules below.

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

Model loads pin an explicit dtype instead of inheriting the checkpoint's configuration:

| Device | Inference dtype |
|---|---|
| CUDA with native bfloat16 support | bfloat16 (emulated support is excluded) |
| Other CUDA devices and MPS | float32 |
| CPU | float32, unless the experimental policy below is enabled |

`CODEDUPES_CPU_BF16=1` enables experimental CPU bfloat16 only when the machine has both a native bf16 ISA (`bf16` on ARM, `amx_bf16`/`avx512_bf16` on x86) and an available mkldnn GEMM backend. The capability check runs at most once per process and persists nothing. `codedupes info` reports the hardware checks and effective policy.

The CPU capability gate does not establish accuracy at the built-in duplicate and search thresholds. Automatic enablement awaits speed and decision-parity validation on supported hardware. TODO before promotion: measure agreement between CPU and CUDA bfloat16 vectors, which currently share a cache namespace, and split their identities if needed.

Both load-time and inference-time CPU fallback reapply this dtype policy. A bfloat16 accelerator model becomes float32 unless the CPU opt-in and capability gate both pass.

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

The CUDA memory cap limits allocator growth; it does not prevent reuse of cached blocks. Lowering it after loading or warming a model can therefore leave inference working, especially with `PYTORCH_ALLOC_CONF=expandable_segments:True`. Clearing unused cache is insufficient when free blocks share an active segment with model weights. The inference-OOM tests run on a fresh CUDA stream that waits for prior model work, forcing an allocation that cannot reuse the original stream's cached blocks. The tests restore the previous memory cap afterward and exercise real allocator errors without changing the configured allocator or simulating exceptions. The load-time OOM test sets the cap before loading weights.

Run the CUDA suite together with the default-model and labeled Rust-fixture GPU smoke tests:

```bash
CODEDUPES_SMOKE_GPU=1 pytest tests/test_semantic_cuda.py tests/test_semantic_smoke.py -m gpu
```

Validated on an NVIDIA GeForce RTX 5090 with PyTorch `2.13.0+cu130` and `expandable_segments:True` on 2026-09-05: all 16 selected tests passed in both normal and reversed collection order, with no CUDA tests skipped.

A companion opt-in smoke test validates every built-in profile against the multi-domain probe corpus in `test_fixtures/search_probes/`: every relevant query must surface its expected function at that profile's default search threshold and every off-topic query must return nothing:

```bash
CODEDUPES_SMOKE_SEARCH=1 pytest tests/test_semantic_smoke.py
```

## Upstream references

- [PyTorch 2.13 release notes](https://pytorch.org/blog/pytorch-2-13-release-blog/)
- [PyTorch 2.13 CUDA streams and memory management](https://docs.pytorch.org/docs/2.13/notes/cuda.html)
- [PyTorch 2.13 MPS backend requirements](https://docs.pytorch.org/docs/2.13/notes/mps.html)
- [PyTorch 2.13 MPS environment variables](https://docs.pytorch.org/docs/2.13/mps_environment_variables.html)
- [PyTorch 2.13 `torch.mps` API](https://docs.pytorch.org/docs/2.13/mps.html)
- [SentenceTransformer device placement](https://sbert.net/docs/package_reference/sentence_transformer/model.html)
- [MLX Metal memory APIs](https://ml-explore.github.io/mlx/build/html/python/metal.html)
- [MLX compiled-function caching](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
