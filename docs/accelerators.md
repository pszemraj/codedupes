# Semantic Accelerators and Apple Silicon

Install the supported runtime and verify MPS availability as described in
[Installation](install.md). The [CLI reference](cli.md) lists device options, and
[model profiles](model-profiles.md) lists model-specific thresholds and tasks.

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

An explicit unavailable accelerator is an error. `codedupes` does not silently reinterpret
`--device mps` as CPU. The only automatic CPU transitions are the documented unsupported-op
and out-of-memory recovery paths below.

## Unsupported MPS operators

PyTorch controls unsupported-operator fallback through
`PYTORCH_ENABLE_MPS_FALLBACK`.

- With `--device mps`, or `--device auto` on macOS, `codedupes` sets the variable to `1`
  before importing PyTorch when the variable is otherwise unset.
- `--mps-fallback` explicitly sets it to `1`.
- `--no-mps-fallback` explicitly sets it to `0`.
- With neither flag, an existing environment value is respected.

Set this before any other code imports PyTorch. If a long-lived Python process has already imported
PyTorch, changing the setting may require a process restart; `codedupes` emits a warning in that
case. The Python API intentionally leaves `PYTORCH_ENABLE_MPS_FALLBACK` at the configured value
because PyTorch reads it as process-wide runtime state; the previous environment value is not
restored after analysis. Library embedders should configure the policy once during process startup
or isolate analyses that need different policies in separate processes.

Unsupported-op fallback is different from out-of-memory recovery. Disabling unsupported-op
fallback does not disable the explicit OOM recovery policy described next.

## MPS memory policy and OOM recovery

No allocator cap is imposed by default. On memory-constrained systems, start with a cap of `0.9`:

```bash
codedupes check ./src --device mps --mps-memory-fraction 0.9
```

The option calls `torch.mps.set_per_process_memory_fraction()` and accepts `(0, 2]`.
`codedupes` rejects `0` because PyTorch defines it as unlimited allocation, which can permit a
system-wide OOM. Values above `1` are accepted for parity with PyTorch but emit a warning because
they exceed the device-recommended working-set size. A cap can cause an earlier, controlled OOM;
it is not a performance setting.

Inference OOM recovery is deterministic:

1. Detach the failed traceback so temporary tensors are no longer retained by Python frames.
2. Log MPS tensor, driver, and recommended-memory statistics when available.
3. Synchronize queued MPS work, run garbage collection, and call `torch.mps.empty_cache()`.
4. Halve the embedding batch size until it reaches one.
5. If an accelerator still OOMs at batch size one, move the cached model to CPU once and retry from
   the originally requested batch size; host memory has different limits, and a CPU OOM re-enters
   the halving ladder above before aborting.

A model-loading MPS OOM has no batch to shrink, so it clears the MPS cache and retries loading once
on CPU. After an MPS-to-CPU OOM fallback, the CPU model remains sticky for that model in a
long-lived process. Call `codedupes.semantic.clear_model_cache()` to force a fresh accelerator
load.

Successful batches do not clear the allocator cache. Embeddings are converted to normalized NumPy
arrays immediately, so pairwise similarity runs on CPU and no large embedding tensor remains
resident in Metal memory.

## Precision and Metal environment variables

The built-in EmbeddingGemma dtype override uses float32 on MPS instead of forcing bfloat16 on Apple
Silicon. CUDA may use bfloat16 when the hardware reports support. Generic HuggingFace models follow
their own model defaults and may not support MPS.

`codedupes` deliberately does not set `PYTORCH_MPS_FAST_MATH` or
`PYTORCH_MPS_PREFER_METAL`. Fast math may change floating-point results around tuned similarity
thresholds, while forcing a particular matmul implementation is a workload-specific optimization.
You can experiment with those variables externally, but re-run the hybrid tuning guardrail and a
representative repository before adopting altered thresholds.

For a native macOS installation, use the default `gte-modernbert-base` profile
first; evaluate `embeddinggemma-300m` only after the default path is stable.

## MLX coexistence

MLX and PyTorch both consume Apple unified memory but manage it separately. `codedupes` never
imports MLX and never touches its allocator; if MLX is already loaded in the process and semantic
execution resolves to MPS, `codedupes` logs one warning about shared unified-memory pressure.
Releasing MLX arrays and clearing MLX caches remains the host application's job.

## Hardware validation

To validate your installed PyTorch wheel and the default model end to end on Apple Silicon
hardware, run the opt-in smoke test:

```bash
CODEDUPES_SMOKE_MPS=1 pytest -m mps tests/test_semantic_smoke.py
```

This test downloads the default model if it is not already cached.

A companion opt-in smoke test validates search quality against the probe corpus in
`test_fixtures/search_probes/`: every relevant query must surface its expected function at the
default search threshold and every off-topic query must return nothing:

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
