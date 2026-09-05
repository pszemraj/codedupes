# Hybrid Confidence Tuning

Tune how gated semantic-only matches are divided between high-confidence and review tiers.

## Guardrail corpus and labels

- Corpus: [`../test_fixtures/hybrid_tuning/crab_visibility`](../test_fixtures/hybrid_tuning/crab_visibility)
- Labels: [`../test_fixtures/hybrid_tuning/labels.json`](../test_fixtures/hybrid_tuning/labels.json)
- Sweep harness: [`../scripts/sweep_hybrid_gates.py`](../scripts/sweep_hybrid_gates.py)
- Semantic threshold harness: [`../scripts/sweep_semantic_thresholds.py`](../scripts/sweep_semantic_thresholds.py)

This corpus is synthetic and tracked for reproducibility.

Use it as a guardrail dataset, not a benchmark.

## Recommended process

1. Run the sweep harness on the tracked synthetic corpus.
2. Rank rows by the high-confidence subset's `f1`, then inspect the separately reported review and total-published counts. Corroboration thresholds must not suppress already-gated semantic matches.
3. Re-validate selected thresholds on at least one real repository before changing defaults.
4. Keep labels/corpus changes explicit in review.

## Run the sweep

```bash
python scripts/sweep_hybrid_gates.py --top-n 15
```

Write a machine-readable report:

```bash
python scripts/sweep_hybrid_gates.py \
  --top-n 25 \
  --json-out scratch/hybrid_sweep_report.json
```

The report embeds the [semantic sweep's calibration manifest](#semantic-threshold-sweep-model-profiles) and records `output_policy: hybrid_high_confidence`: its `tp`/`fp`/`fn` and `precision`/`recall`/`f1` score the published output minus the `semantic_review` tier, unlike the semantic sweep's same-named fields, which score all published pairs. Both sweeps require a model pinned to an immutable 40-character commit.

## Parameter grids

Defaults used by the harness:

- semantic gate: `0.68,0.72,0.76,0.80,0.84,0.88,0.92`
- weak identifier jaccard minimum: `0.10,0.15,0.20,0.25,0.30`
- statement ratio minimum: `0.20,0.25,0.35,0.45,0.55`

Raw semantic candidates are collected at the lowest `--semantic-grid` value; an explicit `--semantic-threshold` above that floor is rejected because grid rows below the collection threshold could never see their pairs.

Override grids as needed:

```bash
python scripts/sweep_hybrid_gates.py \
  --semantic-grid 0.72,0.76,0.80 \
  --weak-jaccard-grid 0.15,0.20,0.25 \
  --statement-ratio-grid 0.25,0.35,0.45
```

The harness uses the analyzer's [hybrid synthesis logic](analysis-defaults.md#hybrid-synthesis-confidence-defaults) and the CLI's model/revision defaults. Each `--semantic-grid` value emulates the [per-language duplicate gate](analysis-defaults.md#semantic-duplicate-gate-defaults) applied before synthesis.

## Semantic threshold sweep (model profiles)

Run the duplicate and search threshold sweeps for built-in model profiles:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/sweep_semantic_thresholds.py --top-n 10
```

By default this sweeps the legacy Python-only `crab_visibility` corpus; its duplicate-threshold report is a guardrail, not the source of the shipped per-language duplicate gates. Those are calibrated from [`../test_fixtures/polyglot_calibration/`](../test_fixtures/polyglot_calibration/README.md), whose README records the per-language re-run command (`--corpus-path`, `--labels-path`, `--language`, and `--duplicate-start`/`--duplicate-stop` to widen the grid below the default floor).

Default report paths:

- [`../test_fixtures/hybrid_tuning/semantic_threshold_report.json`](../test_fixtures/hybrid_tuning/semantic_threshold_report.json) - duplicate thresholds
- [`../test_fixtures/hybrid_tuning/search_threshold_report.json`](../test_fixtures/hybrid_tuning/search_threshold_report.json) - search thresholds, evaluated against [`../test_fixtures/hybrid_tuning/search_probes.json`](../test_fixtures/hybrid_tuning/search_probes.json)

Each report records the full calibration identity per model: the pinned immutable commit, embedding pipeline schema and runtime fingerprint, encode plan (route and prompt per input mode), the requested device plus the effective embedding-space identity the analyzer actually produced (its runtime variant covers dtype and Metal math policy, and reflects an accelerator request that fell back and restarted on CPU - thresholds are never labeled with a device or dtype that did not produce them), embedding dimension, candidate policy, candidate coverage, and SHA-256 digests of the corpus and labels/probes. Duplicate rows score the final combined output, not the raw semantic list. Labels outside the semantic candidate policy can still be recovered by full-scope traditional analysis; the manifest records those recoveries separately and counts only labels reached by neither tier as unreachable. The sweep defaults to the production statement floor and refuses to run for a model that cannot be pinned to a 40-character commit - pass `--model-revision` or pin the profile's `default_revision`. Both grids are overrideable (`--duplicate-start`/`--duplicate-stop`, `--search-start`/`--search-stop`; search default 0.20-0.90), and every manifest records `selected_at_grid_edge`: a boundary selection is censored evidence to re-run with a wider grid, never an interior optimum.

Selection policy is deterministic:

- sort by `f1` (desc), `precision` (desc), `recall` (desc), `fp` (asc), then prefer the looser threshold on remaining ties

Transferring a swept value into a profile default is a reviewed decision, not automatic: re-validate on at least one real repository and follow the [duplicate gate selection policy](analysis-defaults.md#semantic-duplicate-gate-defaults). `tests/test_calibration_reports.py` re-derives each shipped gate from its recorded grid and checks that policy. [Model profiles](model-profiles.md#built-in-profiles) lists search defaults and their calibration evidence.
