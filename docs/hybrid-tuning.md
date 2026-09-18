# Calibrating semantic and hybrid defaults

The [calibration corpus](../test_fixtures/calibration/README.md) contains five
runnable development applications with
reviewed duplicate pairs and independent search relevance for Python, C, Rust,
JavaScript, and TypeScript.

## Checked development-corpus result

The checked [calibration result](../test_fixtures/calibration/calibration-results.json)
records corpus inventory, selected thresholds, metrics, timings, and CPU/MPS
drift. It is authored development data rather than a held-out estimate for
arbitrary repositories. It records no CPU/MPS decision changes for the checked
corpus; normal macOS development should use the default `device=auto`, while CPU
fp32 remains the reproducible calibration reference.

## Reproduce the result

Validate source selectors, pair coverage, tests, and entry points first. The
toolchain-marked pytest integration target and the validator command below run
these behavior commands; they require a stable Rust toolchain, a C compiler and
Make, and Node.js/npm with type-stripping support:

```bash
conda run --name inf pytest -m toolchain
conda run --name inf python scripts/validate_calibration_corpus.py --run-behavior
```

Measure one project, model, and device in each fresh process. Measurements write
to ignored `scratch/calibration/` files and disable embedding-cache reuse.

```bash
conda run --name inf python scripts/measure_calibration.py \
  --project ledger --model gte-modernbert-base --device cpu
conda run --name inf python scripts/measure_calibration.py \
  --project ledger --model gte-modernbert-base --device mps
```

Repeat for every project and built-in model. CPU fp32 is the reference and MPS
is an independent real-device comparison. The measurement, selection, and report
scripts bind artifacts to their source, annotations, model, extraction, runtime,
dtype, batch, and math policy; a material input change requires fresh evidence.
Raw scores and model caches remain local. Calibration rejects
`PYTORCH_MPS_FAST_MATH` because altered Metal arithmetic is not comparable
evidence.

Select per-language duplicate gates and a global top-10 search gate from the CPU
measurements:

```bash
conda run --name inf python scripts/sweep_semantic_thresholds.py \
  --json-out scratch/calibration/threshold-selection.json
```

Once every admission selection is ready, select hybrid visibility constants and
per-language promotion gates:

```bash
conda run --name inf python scripts/sweep_hybrid_gates.py \
  --threshold-selection scratch/calibration/threshold-selection.json \
  --json-out scratch/calibration/hybrid-selection.json
```

Duplicate admission, search, and hybrid visibility discard candidates below 50%
judged precision, maximize judged F1, and prefer higher recall within `0.005` F1
of the optimum. Search preserves explicit no-result probes; duplicate admission
excludes pairs already found by a traditional method. Hybrid selection applies the
precision floor to every language and pooled result while jointly selecting
corroboration and promotion gates.

Run the multi-domain search smoke test against the selected default:

```bash
CODEDUPES_SMOKE_SEARCH=1 conda run --name inf pytest tests/test_semantic_smoke.py -k search
```

Keep these queries unchanged. Each target must rank in the top three without a
score floor, emitted default hits must be relevant, and no-result queries must
stay empty. The checked result records the current floors, search curve, and
metrics.

After applying accepted profile values, rerun both selection commands above so
the recorded current metrics and policy identity match the shipped defaults.
Capture fresh MPS results, then write the compact checked summary:

```bash
conda run --name inf python scripts/report_calibration_distributions.py \
  --json-out test_fixtures/calibration/calibration-results.json
```

This [report script](../scripts/report_calibration_distributions.py) writes the
checked result from the bound raw artifacts and selections. Its companion tests
validate report provenance and arithmetic.
