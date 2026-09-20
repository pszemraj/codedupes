# Calibrating semantic and hybrid defaults

The [calibration corpus](../test_fixtures/calibration/README.md) contains five runnable development applications with reviewed duplicate pairs and independent search relevance for Python, C, Rust, JavaScript, and TypeScript.

## Checked development-corpus result

The checked [calibration result](../test_fixtures/calibration/calibration-results.json) records corpus inventory, selected thresholds, metrics, timings, and CPU/MPS drift. It is authored development data rather than a held-out estimate for arbitrary repositories. It records no CPU/MPS decision changes for the checked corpus; normal macOS development should use the default `device=auto`, while CPU fp32 remains the reproducible calibration reference.

This result completes the development-corpus phase of Issue #20, not the issue. The remaining phase is a compact hand-reviewed real-code check in every supported language, followed by only the targeted corpus additions and recalibration that those checks justify. The current fixtures do not support ecosystem-level statistical claims or cross-language calibration.

The metric tables below are rendered from the checked result by `python scripts/render_calibration_tables.py`; a regression test keeps the published values synchronized with that artifact.

### Selected difficulty recall

Duplicate recall at the selected per-language admission gates is shown as detected / labeled comparable positive pairs. A dash means the language has no labeled hard pair.

| language | GTE easy | GTE medium | GTE hard | Gemma easy | Gemma medium | Gemma hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python | 3/5 | 3/10 | - | 4/5 | 8/10 | - |
| Rust | 5/5 | 5/6 | 0/1 | 5/5 | 5/6 | 0/1 |
| C | 4/5 | 8/14 | - | 5/5 | 13/14 | - |
| JavaScript | 5/5 | 8/14 | - | 4/5 | 8/14 | - |
| TypeScript | 5/5 | 4/6 | 0/1 | 5/5 | 5/6 | 0/1 |
| **Overall** | **22/25** | **28/50** | **0/2** | **23/25** | **39/50** | **0/2** |
| **Recall** | **88.0%** | **56.0%** | **0.0%** | **92.0%** | **78.0%** | **0.0%** |

Difficulty belongs to positive examples, so precision and F1 do not have a difficulty bucket.

### Pooled selected-policy metrics

The duplicate-admission rows include every comparable reviewed pair. The default-visible rows apply the selected hybrid policy to semantic-only findings; deterministic findings are reported separately and excluded from both rows.

| output | model | TP / FP / FN | precision | recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| duplicate admission | GTE | 50 / 7 / 27 | 87.7% | 64.9% | 74.6% |
| duplicate admission | Gemma | 62 / 13 / 15 | 82.7% | 80.5% | 81.6% |
| semantic-only default-visible | GTE | 50 / 2 / 27 | 96.2% | 64.9% | 77.5% |
| semantic-only default-visible | Gemma | 59 / 4 / 18 | 93.7% | 76.6% | 84.3% |
| search | GTE | 72 / 20 / 22 | 78.3% | 76.6% | 77.4% |
| search | Gemma | 72 / 38 / 22 | 65.5% | 76.6% | 70.6% |

The checked JSON contains the exact ratios, per-language rows, candidate windows, hybrid alternatives, runtime metadata, timings, and device-drift evidence behind these rounded tables.

## Historical real-repository smoke snapshot

At commit `9508215`, both profiles were run uncached on live MPS over this repository's production and maintenance Python in `src/` and `scripts/`. Tests, calibration fixtures, local scratch data, generated distributions, and tool metadata were excluded. The scan found no exact, structurally similar, traditional-near, or hybrid-confirmed pair. Manual review of the leading semantic-only results found wrappers, caller/callee pairs, sibling operations, and parallel parser backends rather than code that should be consolidated.

| profile | merge-base policy semantic admissions | policy at `9508215` semantic admissions | merge-base policy default-visible | policy at `9508215` default-visible |
| --- | ---: | ---: | ---: | ---: |
| GTE | 529 | 83 | 139 | 64 |
| Gemma | 245 | 245 | 218 | 145 |

The intermediate policy at `9508215` reduced noise, especially for GTE, but did not meet Issue #20's real-code precision bar. These counts are a historical snapshot, not output from the final phase-one profiles: later source and policy changes make a direct comparison invalid. Similarity-only threshold increases did not cleanly separate fixture positives from related-but-distinct repository functions, while withholding every semantic-only candidate discarded useful recall.

TODO (Issue #20, phase 2): Calibrate default-visible density on versioned, license-recorded real-code pools. Add one `negative_pool` per supported language (use this repository's `src/` and `scripts/` for Python; select independent pinned projects for C, Rust, JavaScript, and TypeScript), capture both built-in models on CPU and live MPS, record each policy's `semantic_high_confidence` pairs per 1,000 embedded units, and set an explicit per-language budget from the observed density-fixture-recall frontier. Pools are unlabeled rate evidence, not false-positive labels, and must not influence fixture precision, recall, or F1. Implement the manifest role, selection constraint, selection/report audit, profile update, and regression coverage as one atomic recalibration after that evidence exists.

## Reproduce the result

Validate source selectors, pair coverage, tests, and entry points first. The toolchain-marked pytest integration target and the validator command below run these behavior commands; they require a stable Rust toolchain, a C compiler and Make, and Node.js/npm with type-stripping support:

```bash
pytest -m toolchain
python scripts/validate_calibration_corpus.py --run-behavior
```

Measure one project, model, and device in each fresh process. Measurements write to ignored `scratch/calibration/` files and disable embedding-cache reuse.

```bash
python scripts/measure_calibration.py \
  --project ledger --model gte-modernbert-base --device cpu
python scripts/measure_calibration.py \
  --project ledger --model gte-modernbert-base --device mps
```

Repeat for every project and built-in model. CPU fp32 is the reference and MPS is an independent real-device comparison. Each run records the corpus embedding device and every probe's query-encoding device; calibration rejects cache reuse or a query fallback away from the requested device while ordinary searches retain automatic fallback. The measurement, selection, and report scripts bind artifacts to their source, annotations, model, extraction, runtime, dtype, batch, and math policy; a material input change requires fresh evidence. Raw scores and model caches remain local. Calibration rejects `PYTORCH_MPS_FAST_MATH` because altered Metal arithmetic is not comparable evidence.

Select per-language duplicate gates and a global top-10 search gate from the CPU measurements:

```bash
python scripts/sweep_semantic_thresholds.py \
  --json-out scratch/calibration/threshold-selection.json
```

Once every admission selection is ready, select hybrid visibility constants and per-language promotion gates:

```bash
python scripts/sweep_hybrid_gates.py \
  --threshold-selection scratch/calibration/threshold-selection.json \
  --json-out scratch/calibration/hybrid-selection.json
```

Duplicate admission, search, and hybrid visibility discard candidates below 50% judged precision, maximize judged F1, and prefer higher recall within `0.005` F1 of the optimum. Search preserves explicit no-result probes; duplicate admission excludes pairs already found by a traditional method. Hybrid selection applies the precision floor to every language and pooled result while jointly selecting corroboration and promotion gates. It keeps promotion disabled when that policy is fixture-equivalent to numeric gates; otherwise it emits the center of each fixture-equivalent numeric plateau rather than its permissive edge. Exact pooled ties prefer the lower identifier-overlap floor and retain a distinct `runner_up` in the checked audit; the phase-two density evaluation determines whether a fixture-equivalent alternative behaves better on real code. Selections bind fixture source, unit and query inputs, model revisions and tasks, plus explicit policy and pipeline versions. Changes to these inputs require fresh evidence; formatting and comments in the calibration implementation do not. New captures record the package version for diagnostics without using it to invalidate otherwise compatible evidence.

Run the multi-domain search smoke test against the selected default:

```bash
CODEDUPES_SMOKE_SEARCH=1 pytest tests/test_semantic_smoke.py -k search
```

Keep these queries unchanged. Each target must rank first without a score floor, emitted default hits must be relevant, and no-result queries must stay empty. The checked result records nearby duplicate and search rows plus the hybrid candidate grids, best-F1 candidate, final-order runner-up (or `null` when no distinct near-best outcome exists), per-language precision evidence, and digests of the exact predicted-pair outcomes so policies with equal aggregate counts remain distinguishable.

The raw-backed report generator reproduces the complete hybrid sweep before it writes those compact audit rows. Without the ignored raw score matrices, the checked-only validator can verify their candidate grids, arithmetic, per-language safety, ordering, outcome-digest distinction, and internally consistent measurement-runtime provenance. It cannot independently prove that a digest names a particular raw prediction set or that no omitted candidate ranked higher. Both paths process saved scores without requiring the reader's runtime, hardware, or math settings to match the capture. Recorded execution metadata remains bound to the raw inputs, and the checked comparison requires a consistent capture runtime across reports. Fresh inference records the runtime that actually produced its scores.

Apply accepted values in [the profile definitions](../src/codedupes/semantic_profiles.py) and [profile tables](model-profiles.md#built-in-profiles) as one change, then run `pytest tests/test_semantic_profiles.py tests/test_calibration.py`. Rerun both selection commands above so the recorded metrics and policy identity match the shipped defaults. Capture fresh MPS results, then write the compact checked summary:

```bash
python scripts/report_calibration_distributions.py \
  --json-out test_fixtures/calibration/calibration-results.json
```

This [report script](../scripts/report_calibration_distributions.py) writes the checked result from the bound raw artifacts and selections. Its companion tests validate report provenance and arithmetic.
