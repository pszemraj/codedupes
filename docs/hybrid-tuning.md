# Calibrating semantic and hybrid defaults

The [calibration corpus](../test_fixtures/calibration/README.md) contains five runnable development applications with reviewed duplicate pairs and independent search relevance for Python, C, Rust, JavaScript, and TypeScript.

## Checked development-corpus result

The checked [calibration result](../test_fixtures/calibration/calibration-results.json) records corpus inventory, selected thresholds, metrics, timings, and CPU/MPS drift. It is authored development data rather than a held-out estimate for arbitrary repositories. It records no CPU/MPS decision changes for the checked corpus; normal macOS development should use the default `device=auto`, while CPU fp32 remains the reproducible calibration reference.

This result completes the development-corpus phase of Issue #20, not the issue. The remaining phase is a compact hand-reviewed real-code check in every supported language, followed by only the targeted corpus additions and recalibration that those checks justify. The current fixtures do not support ecosystem-level statistical claims or cross-language calibration.

### Selected difficulty recall

Duplicate recall at the selected per-language admission gates is shown as detected / labeled comparable positive pairs. A dash means the language has no labeled hard pair.

| language | GTE easy | GTE medium | GTE hard | Gemma easy | Gemma medium | Gemma hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python | 3/5 | 3/10 | - | 4/5 | 8/10 | - |
| Rust | 5/5 | 5/6 | 0/1 | 5/5 | 5/6 | 0/1 |
| C | 4/5 | 8/14 | - | 5/5 | 13/14 | - |
| JavaScript | 4/5 | 8/14 | - | 4/5 | 8/14 | - |
| TypeScript | 5/5 | 4/6 | 0/1 | 5/5 | 5/6 | 0/1 |
| **Overall** | **21/25** | **28/50** | **0/2** | **23/25** | **39/50** | **0/2** |
| **Recall** | **84.0%** | **56.0%** | **0.0%** | **92.0%** | **78.0%** | **0.0%** |

Difficulty belongs to positive examples, so precision and F1 do not have a difficulty bucket.

### Pooled selected-policy metrics

The duplicate-admission rows include every comparable reviewed pair. The default-visible rows apply the selected hybrid policy to semantic-only findings; deterministic findings are reported separately and excluded from both rows.

| output | model | TP / FP / FN | precision | recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| duplicate admission | GTE | 49 / 5 / 28 | 90.7% | 63.6% | 74.8% |
| duplicate admission | Gemma | 62 / 13 / 15 | 82.7% | 80.5% | 81.6% |
| semantic-only default-visible | GTE | 49 / 1 / 28 | 98.0% | 63.6% | 77.2% |
| semantic-only default-visible | Gemma | 59 / 4 / 18 | 93.7% | 76.6% | 84.3% |
| search | GTE | 72 / 20 / 22 | 78.3% | 76.6% | 77.4% |
| search | Gemma | 69 / 32 / 25 | 68.3% | 73.4% | 70.8% |

The checked JSON contains the exact ratios, per-language rows, candidate windows, hybrid alternatives, runtime metadata, timings, and device-drift evidence behind these rounded tables.

## Historical real-repository smoke snapshot

At commit `9508215`, both profiles were run uncached on live MPS over this repository's production and maintenance Python in `src/` and `scripts/`. Tests, calibration fixtures, local scratch data, generated distributions, and tool metadata were excluded. The scan found no exact, structurally similar, traditional-near, or hybrid-confirmed pair. Manual review of the leading semantic-only results found wrappers, caller/callee pairs, sibling operations, and parallel parser backends rather than code that should be consolidated.

| profile | merge-base semantic admissions | phase-one semantic admissions | merge-base default-visible | phase-one default-visible |
| --- | ---: | ---: | ---: | ---: |
| GTE | 529 | 83 | 139 | 64 |
| Gemma | 245 | 245 | 218 | 145 |

The phase-one policy reduced noise, especially for GTE, but did not meet Issue #20's real-code precision bar. Similarity-only threshold increases did not cleanly separate fixture positives from related-but-distinct repository functions, while withholding every semantic-only candidate discarded useful recall. The next real-code pass should add a compact representative negative slice, recalibrate, and repeat the check across all five languages rather than add repository-specific suppression.

## Reproduce the result

Validate source selectors, pair coverage, tests, and entry points first. The toolchain-marked pytest integration target and the validator command below run these behavior commands; they require a stable Rust toolchain, a C compiler and Make, and Node.js/npm with type-stripping support:

```bash
conda run --name inf pytest -m toolchain
conda run --name inf python scripts/validate_calibration_corpus.py --run-behavior
```

Measure one project, model, and device in each fresh process. Measurements write to ignored `scratch/calibration/` files and disable embedding-cache reuse.

```bash
conda run --name inf python scripts/measure_calibration.py \
  --project ledger --model gte-modernbert-base --device cpu
conda run --name inf python scripts/measure_calibration.py \
  --project ledger --model gte-modernbert-base --device mps
```

Repeat for every project and built-in model. CPU fp32 is the reference and MPS is an independent real-device comparison. The measurement, selection, and report scripts bind artifacts to their source, annotations, model, extraction, runtime, dtype, batch, and math policy; a material input change requires fresh evidence. Raw scores and model caches remain local. Calibration rejects `PYTORCH_MPS_FAST_MATH` because altered Metal arithmetic is not comparable evidence.

Select per-language duplicate gates and a global top-10 search gate from the CPU measurements:

```bash
conda run --name inf python scripts/sweep_semantic_thresholds.py \
  --json-out scratch/calibration/threshold-selection.json
```

Once every admission selection is ready, select hybrid visibility constants and per-language promotion gates:

```bash
conda run --name inf python scripts/sweep_hybrid_gates.py \
  --threshold-selection scratch/calibration/threshold-selection.json \
  --json-out scratch/calibration/hybrid-selection.json
```

Duplicate admission, search, and hybrid visibility discard candidates below 50% judged precision, maximize judged F1, and prefer higher recall within `0.005` F1 of the optimum. Search preserves explicit no-result probes; duplicate admission excludes pairs already found by a traditional method. Hybrid selection applies the precision floor to every language and pooled result while jointly selecting corroboration and promotion gates. Selections bind an explicit algorithm version, objective, search depth, and candidate grids rather than source-file bytes. Formatting, comments, and model aliases therefore do not invalidate evidence; changing a recorded policy value does.

Run the multi-domain search smoke test against the selected default:

```bash
CODEDUPES_SMOKE_SEARCH=1 conda run --name inf pytest tests/test_semantic_smoke.py -k search
```

Keep these queries unchanged. Each target must rank first without a score floor, emitted default hits must be relevant, and no-result queries must stay empty. The checked result records nearby duplicate and search rows plus the hybrid candidate grids, best-F1 candidate, final-order runner-up, and per-language precision evidence.

The raw-backed report generator reproduces the complete hybrid sweep before it writes those compact audit rows. Without the ignored raw score matrices, the checked-only validator can verify their candidate grids, arithmetic, per-language safety, and ordering claims, but it cannot independently prove that no omitted candidate ranked higher.

Apply accepted values in [the profile definitions](../src/codedupes/semantic_profiles.py) and [profile tables](model-profiles.md#built-in-profiles) as one change, then run `conda run --name inf pytest tests/test_semantic_profiles.py tests/test_calibration.py`. Rerun both selection commands above so the recorded metrics and policy identity match the shipped defaults. Capture fresh MPS results, then write the compact checked summary:

```bash
conda run --name inf python scripts/report_calibration_distributions.py \
  --json-out test_fixtures/calibration/calibration-results.json
```

This [report script](../scripts/report_calibration_distributions.py) writes the checked result from the bound raw artifacts and selections. Its companion tests validate report provenance and arithmetic.
