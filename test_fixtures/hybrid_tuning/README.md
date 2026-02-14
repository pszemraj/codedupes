# Hybrid gate tuning fixtures

This directory contains a tracked synthetic corpus used to tune and regression-check
hybrid semantic-only gate thresholds. The source-of-truth workflow is documented in
[docs/hybrid-tuning.md](https://github.com/pszemraj/codedupes/blob/main/docs/hybrid-tuning.md).

## Scope

- Corpus root: [`test_fixtures/hybrid_tuning/crab_visibility`](https://github.com/pszemraj/codedupes/tree/main/test_fixtures/hybrid_tuning/crab_visibility)
- Labels: [`test_fixtures/hybrid_tuning/labels.json`](https://github.com/pszemraj/codedupes/blob/main/test_fixtures/hybrid_tuning/labels.json)
- Sweep harness: [`scripts/sweep_hybrid_gates.py`](https://github.com/pszemraj/codedupes/blob/main/scripts/sweep_hybrid_gates.py)

## Best-practice boundaries

- This corpus is a synthetic guardrail dataset, not a quality benchmark.
- Do not tune thresholds from this corpus alone; validate against at least one real repository.
- Keep labels explicit and reviewable in `labels.json`.
- Keep corpus deterministic and free from generated artifacts.
