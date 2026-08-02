# Hybrid Gate Tuning

Tune hybrid semantic-only gates for high precision while preserving recall on known good pairs.

## Guardrail corpus and labels

- Corpus: [`../test_fixtures/hybrid_tuning/crab_visibility`](../test_fixtures/hybrid_tuning/crab_visibility)
- Labels: [`../test_fixtures/hybrid_tuning/labels.json`](../test_fixtures/hybrid_tuning/labels.json)
- Sweep harness: [`../scripts/sweep_hybrid_gates.py`](../scripts/sweep_hybrid_gates.py)
- Semantic threshold harness: [`../scripts/sweep_semantic_thresholds.py`](../scripts/sweep_semantic_thresholds.py)

This corpus is synthetic and tracked for reproducibility.

Use it as a guardrail dataset, not a benchmark.

## Recommended process

1. Run the sweep harness on the tracked synthetic corpus.
2. Select top candidate rows by `f1`, then prefer higher precision if tied.
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

## Parameter grids

Defaults used by the harness:

- semantic-only minimum: `0.85,0.88,0.90,0.92,0.94`
- weak identifier jaccard minimum: `0.10,0.15,0.20,0.25,0.30`
- statement ratio minimum: `0.20,0.25,0.35,0.45,0.55`

Override grids as needed:

```bash
python scripts/sweep_hybrid_gates.py \
  --semantic-grid 0.88,0.90,0.92 \
  --weak-jaccard-grid 0.15,0.20,0.25 \
  --statement-ratio-grid 0.25,0.35,0.45
```

The harness uses the same analyzer synthesis logic and model/revision defaults as the CLI, so sweep results transfer directly to production gate values.

## Semantic threshold sweep (model profiles)

Run the duplicate and search threshold sweeps for built-in model profiles:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/sweep_semantic_thresholds.py --top-n 10
```

Default report paths:

- [`../test_fixtures/hybrid_tuning/semantic_threshold_report.json`](../test_fixtures/hybrid_tuning/semantic_threshold_report.json) — duplicate thresholds
- [`../test_fixtures/hybrid_tuning/search_threshold_report.json`](../test_fixtures/hybrid_tuning/search_threshold_report.json) — search thresholds, evaluated against [`../test_fixtures/hybrid_tuning/search_probes.json`](../test_fixtures/hybrid_tuning/search_probes.json)

Each report records the full calibration identity per model: the pinned immutable commit, embedding pipeline schema and runtime fingerprint, encode plan (route and prompt per input mode), device and dtype, embedding dimension, candidate policy, and SHA-256 digests of the corpus and labels/probes. The sweep refuses to run for a model that cannot be pinned to a 40-character commit — pass `--model-revision` or pin the profile's `default_revision`.

Selection policy is deterministic:

- sort by `f1` (desc), `precision` (desc), `recall` (desc), `fp` (asc), then prefer the looser threshold on remaining ties

Transferring a swept value into a profile default is a reviewed decision, not automatic: re-validate on at least one real repository, and prefer recall when stepping off the F1-best row. Production gate values are listed in [Analysis defaults](analysis-defaults.md). Model-specific semantic thresholds are listed in [Model profiles](model-profiles.md).
