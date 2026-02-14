# Hybrid Gate Tuning (Best-Practice Workflow)

This page defines the recommended workflow for tuning hybrid semantic-only gates.

Primary goal: keep hybrid output high-precision while preserving recall on known good pairs.

## Guardrail corpus and labels

- Corpus: [`test_fixtures/hybrid_tuning/crab_visibility`](https://github.com/pszemraj/codedupes/tree/main/test_fixtures/hybrid_tuning/crab_visibility)
- Labels: [`test_fixtures/hybrid_tuning/labels.json`](https://github.com/pszemraj/codedupes/blob/main/test_fixtures/hybrid_tuning/labels.json)
- Sweep harness: [`scripts/sweep_hybrid_gates.py`](https://github.com/pszemraj/codedupes/blob/main/scripts/sweep_hybrid_gates.py)
- Semantic threshold harness: [`scripts/sweep_semantic_thresholds.py`](https://github.com/pszemraj/codedupes/blob/main/scripts/sweep_semantic_thresholds.py)

This corpus is synthetic and tracked for reproducibility.

Important boundary: this is a guardrail dataset, not a benchmark.

## Recommended process

1. Run the sweep harness on the tracked synthetic corpus.
2. Select top candidate rows by `f1`, then prefer higher precision if tied.
3. Re-validate selected thresholds on at least one real repository before changing defaults.
4. Keep labels/corpus changes explicit in review.

## Run the sweep

```bash
conda run --name inf python scripts/sweep_hybrid_gates.py --top-n 15
```

Write a machine-readable report:

```bash
conda run --name inf python scripts/sweep_hybrid_gates.py \
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
conda run --name inf python scripts/sweep_hybrid_gates.py \
  --semantic-grid 0.88,0.90,0.92 \
  --weak-jaccard-grid 0.15,0.20,0.25 \
  --statement-ratio-grid 0.25,0.35,0.45
```

## Model/runtime notes

- Harness uses the same analyzer synthesis logic as production.
- By default it uses the same model/revision defaults as the CLI.
- Keep runtime metadata (model, revision, dependency versions) when recording decisions.

## Semantic threshold sweep (model profiles)

Run the semantic threshold sweep for built-in model profiles:

```bash
CUDA_VISIBLE_DEVICES='' conda run --name inf python scripts/sweep_semantic_thresholds.py --top-n 10
```

Default report path:

- [`test_fixtures/hybrid_tuning/semantic_threshold_report.json`](https://github.com/pszemraj/codedupes/blob/main/test_fixtures/hybrid_tuning/semantic_threshold_report.json)

Selection policy is deterministic:

- sort by `f1` (desc), `precision` (desc), `recall` (desc), `fp` (asc)

## Current defaults

Current production gate defaults are defined in
[`src/codedupes/analyzer.py`](https://github.com/pszemraj/codedupes/blob/main/src/codedupes/analyzer.py):

- semantic-only minimum: `0.92`
- weak identifier jaccard minimum: `0.20`
- statement ratio minimum: `0.35`

Current model-profile semantic thresholds are defined in
[docs/model-profiles.md](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md).
