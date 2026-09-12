# Semantic accelerators and Apple Silicon

Install the supported runtime and verify MPS availability as described in [Installation](install.md). The [CLI reference](cli.md) lists device options, and [model profiles](model-profiles.md) lists model-specific thresholds and tasks.

Device selection applies only to semantic embedding; traditional-only analysis does not load a model or initialize PyTorch. Use `--device cpu` when you need to avoid accelerator use, and use an explicit accelerator only to require that hardware.

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

## Accelerator OOM recovery and MPS memory policy

No MPS allocator cap is imposed by default. On memory-constrained systems, start with a cap of `0.9`:

```bash
codedupes check ./src --device mps --mps-memory-fraction 0.9
```

The option calls `torch.mps.set_per_process_memory_fraction()` and accepts `(0, 2]`. `codedupes` rejects `0` because PyTorch defines it as unlimited allocation, which can permit a system-wide OOM. Values above `1` are accepted for parity with PyTorch but emit a warning because they exceed the device-recommended working-set size. A cap can cause an earlier, controlled OOM; it is not a performance setting. The setting is process-global: after codedupes applies a custom cap, the next run whose configuration leaves the option unset restores the allocator baseline captured from `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, or PyTorch's `1.7` default when the environment is unset - including fully cache-covered runs and warm query hits, which never prepare a device. `clear_model_cache()` releases weights but does not itself change allocator policy.

CUDA and MPS inference use the same deterministic OOM recovery ladder. An MPS `Invalid buffer size` failure - a single tensor above Metal's per-buffer cap, raised without any "out of memory" phrase - also enters that ladder:

1. Detach the failed traceback so temporary tensors are no longer retained by Python frames.
2. Log one warning per failed attempt, including MPS tensor, driver, and recommended-memory statistics when available.
3. Synchronize queued MPS work, run garbage collection, and call `torch.mps.empty_cache()`.
4. Halve the embedding batch size until it reaches one.
5. If an accelerator still OOMs at batch size one, move the cached model to CPU once and retry from the originally requested batch size capped at 32 (`CPU_FALLBACK_MAX_BATCH_SIZE`); host memory has different limits, but host OOM can arrive as an uncatchable OOM-killer kill rather than a Python exception, so an accelerator-sized request (say 512) never carries over. A catchable CPU OOM re-enters the halving ladder above before aborting. The move reapplies the [CPU dtype policy](#precision-and-metal-environment-variables).

A model-loading accelerator OOM has no batch to shrink, so it clears that device's cache and retries loading once on CPU. After an accelerator-to-CPU OOM fallback, the CPU model remains sticky for that model in a long-lived process. Call `codedupes.semantic.clear_model_cache()` to force a fresh accelerator load.

Successful batches do not clear the allocator cache. Embeddings are converted to normalized NumPy arrays immediately, so pairwise similarity runs on CPU and no large embedding tensor remains resident in Metal memory.

Fresh embeddings must have the expected shape and row count. Non-finite or zero accelerator output retries once on CPU using the same capped batch policy; invalid CPU output fails. Valid rows become unit-normalized float32 arrays, making dot products cosine similarities. Cache-row repair is described under [storage and consistency](caching.md#storage-and-consistency).

## Precision and Metal environment variables

Model loads pin an explicit dtype instead of inheriting the checkpoint's configuration:

| Device | Inference dtype |
| --- | --- |
| CUDA with native bfloat16 support | bfloat16 (emulated support is excluded) |
| Other CUDA devices and MPS | float32 |
| CPU | float32, unless the experimental policy below is enabled |

`CODEDUPES_CPU_BF16=1` enables experimental CPU bfloat16 only when the machine has both a native bf16 ISA (`bf16` on ARM, `amx_bf16`/`avx512_bf16` on x86) and an available mkldnn GEMM backend. The capability check runs at most once per process and persists nothing. `codedupes info --verbose` reports the hardware checks and effective policy.

The CPU capability gate does not establish accuracy at the built-in duplicate and search thresholds. Automatic enablement awaits speed and decision-parity validation on supported hardware. TODO before promotion: measure agreement between CPU and CUDA bfloat16 vectors, which currently share a cache namespace, and split their identities if needed.

Both load-time and inference-time CPU fallback reapply this dtype policy. A bfloat16 accelerator model becomes float32 unless the CPU opt-in and capability gate both pass.

A run keyed under a non-default (bfloat16) dtype variant whose live execution can no longer produce bfloat16 - an accelerator OOM cast the model to float32 mid-run - discards any cache hits recorded under that key and recomputes the whole corpus in one coherent policy, mirroring the fast-math precedent below; a write that would otherwise land in the wrong key space is skipped instead, costing a cache miss next run rather than a poisoned key space.

The restarted corpus records its faithful CPU identity and stays directly searchable: queries follow that recorded policy even while the analyzer still requests the accelerator. Conversely, a query whose own encode falls back and casts to float32 against a corpus still keyed bfloat16 aborts before the similarity comparison - the correctness boundary is the dot product, not just the cache key.

`codedupes` deliberately does not set `PYTORCH_MPS_FAST_MATH` or `PYTORCH_MPS_PREFER_METAL`. Fast math may change floating-point results around tuned similarity thresholds, while forcing a particular matmul implementation is workload-specific. You can experiment with those variables externally, but re-run the hybrid tuning guardrail and a representative repository before adopting altered thresholds. Changing fast math re-embeds MPS-capable requests; if execution then leaves MPS, the corpus restarts under the effective CPU policy and an incompatible standalone query aborts before comparison. `PYTORCH_MPS_PREFER_METAL` selects among faithful float32 implementations and shares their identity. See [cache runtime identity](caching.md#runtime-identity) for key composition and reuse boundaries.

For a native macOS installation, use the default `gte-modernbert-base` profile first; evaluate `embeddinggemma-300m` only after the default path is stable.

## MLX coexistence

MLX and PyTorch both consume Apple unified memory but manage it separately. `codedupes` never imports MLX and never touches its allocator; if MLX is already loaded in the process and semantic execution resolves to MPS, `codedupes` logs one warning about shared unified-memory pressure. Releasing MLX arrays and clearing MLX caches remains the host application's job.

## Hardware validation

This section is for contributors validating accelerator support; normal use only needs `codedupes info` and the default `--device auto`. The tests run against real hardware and need the default model already available locally. Run a semantic check first to fetch it.

MPS validation runs only where PyTorch reports a usable MPS device; otherwise the suite skips:

```bash
pytest tests/test_semantic_mps.py
```

CUDA validation likewise runs only where `torch.cuda.is_available()` is true:

```bash
pytest tests/test_semantic_cuda.py
```

Both suites check inference against the requested device, cache behavior, and CPU recovery. They intentionally provoke real accelerator OOMs to exercise the recovery ladder, so do not run them alongside a workload that needs the GPU/Metal memory.

The optional CUDA smoke command also exercises the default model and labeled Rust fixture:

```bash
CODEDUPES_SMOKE_GPU=1 pytest tests/test_semantic_cuda.py tests/test_semantic_smoke.py -m gpu
```

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
