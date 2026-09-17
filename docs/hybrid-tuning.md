# Calibrating semantic and hybrid defaults

Issue #20 replaces the old synthetic inputs with five runnable development
applications. The [corpus contract](../test_fixtures/calibration/README.md)
contains reviewed easy, medium, and hard duplicate pairs plus independent search
relevance for Python, C, Rust, JavaScript, and TypeScript.

Validate source selectors, pair coverage, tests, and entry points first:

```bash
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

Repeat for `ledger`, `cowsay`, `c_metering`, `javascript`, and `harbor-ts`, and
for `gte-modernbert-base` and `embeddinggemma-300m`. CPU fp32 is the reference;
MPS is an independent real-device comparison. A source, model, extraction, task,
prompt, or dtype-policy change makes an artifact stale. Label and threshold
edits reuse its raw scores.

Keeping CPU as the calibration reference is a reproducibility convention, not a
separate runtime threshold policy. The checked Issue #20 comparison found a
maximum CPU/MPS score drift of `7.75e-7` and no duplicate or search decision
changes for either built-in model. Treat CPU fp32 and MPS fp32 as functionally
equivalent for these profiles, and use the default `device=auto` during normal
macOS development so Apple silicon selects the faster MPS path. Continue to run
the independent CPU/MPS comparison when source, model, extraction, task, prompt,
or dtype policy changes.

Select per-language duplicate gates and the global top-10 search gate from the
CPU measurements:

```bash
conda run --name inf python scripts/sweep_semantic_thresholds.py \
  --json-out scratch/calibration/threshold-selection.json
```

If the output reports unjudged or ambiguous predictions, review those source
pairs and rerun the sweep. Once every admission selection is ready, select the
hybrid visibility constants and per-language promotion gates:

```bash
conda run --name inf python scripts/sweep_hybrid_gates.py \
  --threshold-selection scratch/calibration/threshold-selection.json \
  --json-out scratch/calibration/hybrid-selection.json
```

The selectors maximize judged F1, then prefer fewer unresolved predictions,
higher recall, and higher precision. Exact ties use a stable midpoint. Reports
keep reviewed ambiguities and unjudged findings separate from judged-only
precision, and break positive recall out by authored difficulty.

After applying the selected profile values, capture fresh MPS results and write
the compact checked summary:

```bash
conda run --name inf python scripts/report_calibration_distributions.py \
  --json-out test_fixtures/calibration/calibration-results.json
```

The summary records timings, effective devices, replay parity, score drift, and
duplicate/search decision changes. Raw score matrices and model caches stay out
of Git.
