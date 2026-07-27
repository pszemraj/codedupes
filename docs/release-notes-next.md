# Next Release Engineering Notes

This document records the implementation and validation scope for the next release. It is not a
replacement for the CLI, output, model-profile, or accelerator source-of-truth documents.

## Breaking baseline change

- Minimum PyTorch version is now `2.13.0`; the supported range is `>=2.13.0,<3`. Dev and rc builds
  of an in-range release (for example `2.13.0.dev20250101`) are accepted.
- The transformers pin widened from `<5` to `>=4.51,<6`; validated against transformers 5.14.1 with
  sentence-transformers 5.6.1.
- `packaging` is now an explicit runtime dependency because semantic compatibility checks import it
  directly.
- Semantic startup now rejects an unsupported PyTorch runtime even when dependency resolution was
  bypassed with `--no-deps` or an editable/source-only environment.
- VCS-less source snapshots now build with an explicit `0.0.0+unknown` fallback; tagged Git builds
  continue to derive the release version from VCS metadata. The sdist now includes tests, docs,
  scripts, and tuning fixtures.

## Semantic runtime changes

- Added explicit `auto`, `cpu`, `cuda`, and `mps` selection for both CLI workflows and the Python
  API.
- Added pre-import MPS unsupported-operator fallback control; `codedupes info` configures the
  fallback environment before importing torch so diagnostics do not pin the setting for later
  same-process runs.
- Added an optional, validated MPS per-process memory fraction. The fraction is ignored (with a
  log message) when the resolved device is not MPS, so `--device auto --mps-memory-fraction X` is
  safe on CPU-only and CUDA hosts.
- Made model caching device-aware and serialized model lifecycle/inference across threads.
- Added MPS memory diagnostics, synchronized allocator cleanup, adaptive batch-size OOM retries,
  and one final CPU retry. The CPU retry restarts from the originally requested batch size and
  re-enters the halving ladder if host memory also OOMs.
- Made MPS-to-CPU OOM fallback state explicit so later calls do not falsely report MPS execution;
  reusing the fallback model for an accelerator request warns once per fallback event.
- Gave `search` its own default threshold. Model profiles now carry a search threshold
  (`0.50` for `gte-modernbert-base`, calibrated against real corpora on Apple Silicon) separate
  from the duplicate threshold, so default searches return ranked matches instead of nothing;
  explicit `--threshold`/`--semantic-threshold` overrides still win.
- Kept semantic embeddings as normalized NumPy arrays immediately after inference.
- Added MLX coexistence diagnostics without importing or mutating the MLX allocator.
- Avoided forced MPS fast math, forced Metal matmul selection, and success-path cache clearing.

## Static-analysis fixes

- `ast.NodeVisitor` and `ast.NodeTransformer` `visit_*` hooks are no longer reported as unused when
  their containing class is proven to inherit from the visitor hierarchy, including local
  subclasses.
- An unrelated ordinary method named `visit_*` remains eligible for unused-code analysis.
- Absolute paths passed without the required `check` or `search` command now produce a CLI usage
  error instead of help with exit code zero.

## Validation boundary

The offline suite covers deterministic MPS simulations and ordinary regressions. Physical Apple
Silicon validation is an explicit release step:

```bash
CODEDUPES_SMOKE_MPS=1 pytest -m mps tests/test_semantic_smoke.py
```

For this release it was performed on an Apple M5 (32 GB unified memory, macOS Tahoe, PyTorch
2.13.0): the smoke test passed with the model on `mps`, a full `codedupes check` of this repo ran
roughly 38x faster on MPS than on CPU with bit-identical duplicate scores across the two devices,
and strict `--no-mps-fallback` runs completed without hitting unsupported operators.
