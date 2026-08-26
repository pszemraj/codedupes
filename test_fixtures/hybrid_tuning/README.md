# Hybrid gate tuning fixtures

This directory contains the synthetic corpus and labels used by the [hybrid gate tuning workflow](../../docs/hybrid-tuning.md).

## Contents

- [`crab_visibility`](crab_visibility): deterministic Python corpus
- [`labels.json`](labels.json): expected duplicate groups
- [`search_probes.json`](search_probes.json): labeled natural-language search probes
- [`semantic_threshold_report.json`](semantic_threshold_report.json): legacy Python-only duplicate-threshold sweep report with per-model calibration manifests; the shipped per-language duplicate gates are calibrated from [`test_fixtures/polyglot_calibration/`](../polyglot_calibration/README.md) instead
- [`search_threshold_report.json`](search_threshold_report.json): search-threshold sweep report with per-model calibration manifests

## Constraints

- Do not tune thresholds from this corpus alone; validate against at least one real repository.
- Keep corpus deterministic and free from generated artifacts.
- Regenerate both reports with `scripts/sweep_semantic_thresholds.py` whenever the corpus, labels, probes, pinned model commits, or embedding pipeline change; the recorded manifests are the reproducibility contract for the search-threshold defaults. The per-language duplicate gates shipped in `codedupes.semantic_profiles` are calibrated from [`test_fixtures/polyglot_calibration/`](../polyglot_calibration/README.md), not from this corpus.
