# Semantic Accelerators and Apple Silicon

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

An explicit unavailable accelerator is an error. `codedupes` does not silently reinterpret `--device mps` as CPU, and the check applies even when a warm embedding cache makes inference unnecessary. The only automatic CPU transitions are the documented unsupported-op and out-of-memory recovery paths below.

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

The option calls `torch.mps.set_per_process_memory_fraction()` and accepts `(0, 2]`. `codedupes` rejects `0` because PyTorch defines it as unlimited allocation, which can permit a system-wide OOM. Values above `1` are accepted for parity with PyTorch but emit a warning because they exceed the device-recommended working-set size. A cap can cause an earlier, controlled OOM; it is not a performance setting.

Inference OOM recovery is deterministic. An MPS `Invalid buffer size` failure - a single tensor above Metal's per-buffer cap, raised without any "out of memory" phrase - classifies as MPS OOM and recovers through the same ladder:

1. Detach the failed traceback so temporary tensors are no longer retained by Python frames.
2. Log one warning per failed attempt, including MPS tensor, driver, and recommended-memory statistics when available.
3. Synchronize queued MPS work, run garbage collection, and call `torch.mps.empty_cache()`.
4. Halve the embedding batch size until it reaches one.
5. If an accelerator still OOMs at batch size one, move the cached model to CPU once and retry from the originally requested batch size; host memory has different limits, and a CPU OOM re-enters the halving ladder above before aborting. The fallback deliberately keeps the model's load-time dtype (including CUDA-selected bfloat16) to preserve the accepted numeric format and reduce memory pressure on this last-resort path.

A model-loading MPS OOM has no batch to shrink, so it clears the MPS cache and retries loading once on CPU. After an MPS-to-CPU OOM fallback, the CPU model remains sticky for that model in a long-lived process. Call `codedupes.semantic.clear_model_cache()` to force a fresh accelerator load.

Successful batches do not clear the allocator cache. Embeddings are converted to normalized NumPy arrays immediately, so pairwise similarity runs on CPU and no large embedding tensor remains resident in Metal memory.

## Precision and Metal environment variables

Model loads pin an explicit dtype instead of inheriting the checkpoint's config-declared one: bfloat16 on CUDA hardware with native bf16 support (Ampere or newer; pre-Ampere emulated bf16 is excluded), float32 on CPU and MPS, for every model family. Without the pin, Transformers 5's `dtype="auto"` default runs float16-configured checkpoints (including the default `gte-modernbert-base`) in half precision - about 10x slower on CPU and off the faithful-float32 tolerance. CPU and MPS stay float32 by measurement, not caution: on Apple Silicon, CPU bfloat16 is emulated (~16x slower) and MPS bfloat16 gains only ~13% runtime while drifting pair similarities ~1e-2 (tuned-threshold scale) and cold-splitting the shared CPU/MPS cache key space.

`codedupes` deliberately does not set `PYTORCH_MPS_FAST_MATH` or `PYTORCH_MPS_PREFER_METAL`. Fast math may change floating-point results around tuned similarity thresholds, while forcing a particular matmul implementation is a workload-specific optimization. You can experiment with those variables externally, but re-run the hybrid tuning guardrail and a representative repository before adopting altered thresholds. The persistent embedding cache keys `PYTORCH_MPS_FAST_MATH` into its vector identity whenever the request could execute on MPS (explicit `mps`, or `auto` on macOS), so toggling the policy re-embeds instead of serving vectors from the other math mode; the key mirrors torch's exact rule, where any set value except the literal `0` enables fast math (an empty string enables it). A keyed run that actually executes off MPS - accelerator unavailable, or the OOM fallback landed on CPU - skips publishing its vectors, keeping the two spaces unmixed. `PYTORCH_MPS_PREFER_METAL` (presence-only: setting it to `0` still enables it) only selects among faithful float32 implementations and intentionally shares the key space, like CPU and MPS float32 do.

For a native macOS installation, use the default `gte-modernbert-base` profile first; evaluate `embeddinggemma-300m` only after the default path is stable.

## MLX coexistence

MLX and PyTorch both consume Apple unified memory but manage it separately. `codedupes` never imports MLX and never touches its allocator; if MLX is already loaded in the process and semantic execution resolves to MPS, `codedupes` logs one warning about shared unified-memory pressure. Releasing MLX arrays and clearing MLX caches remains the host application's job.

## Hardware validation

MPS behavior is validated on real Apple Silicon hardware only - the test suite contains no simulated MPS. `tests/test_semantic_mps.py` runs automatically wherever PyTorch reports a usable MPS device and skips only where the hardware is genuinely absent (a non-Mac host, or a sandbox that blocks Metal device access - a skipped run performs zero MPS validation, so run it from an environment with device access):

```bash
pytest tests/test_semantic_mps.py
```

The suite loads the pinned default model on `mps`, checks CPU/MPS embedding parity, validates explicit `--device mps` requests against a warm embedding cache, and provokes genuine Metal allocator out-of-memory (by lowering `torch.mps.set_per_process_memory_fraction`) to prove load-time CPU fallback, the batch-halving ladder, and query-encode recovery work on the real allocator. The default model must already be cached locally (any prior `codedupes check` or `hf download` does this).

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
