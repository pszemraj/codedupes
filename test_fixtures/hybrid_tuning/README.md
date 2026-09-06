# Hybrid gate tuning fixtures

This directory contains the synthetic corpus and labels used by the [hybrid gate tuning workflow](../../docs/hybrid-tuning.md).

It is a calibration guardrail, not an end-user sample project. Run tuning commands from the repository root and write exploratory JSON under `scratch/`; compare it with the recorded reports before making a deliberate calibration change.

## Contents

- [`crab_visibility`](crab_visibility): deterministic Python corpus
- [`labels.json`](labels.json): expected duplicate groups
- [`search_probes.json`](search_probes.json): labeled natural-language search probes
- [`semantic_threshold_report.json`](semantic_threshold_report.json): legacy Python-only duplicate-threshold sweep report with per-model calibration manifests; the shipped per-language duplicate gates are calibrated from [`test_fixtures/polyglot_calibration/`](../polyglot_calibration/README.md) instead
- [`search_threshold_report.json`](search_threshold_report.json): search-threshold sweep report with per-model calibration manifests

## Constraints

- Keep corpus deterministic and free from generated artifacts.
- Re-run the [semantic threshold sweep](../../docs/hybrid-tuning.md#semantic-threshold-sweep-model-profiles) whenever the corpus, labels, probes, pinned model commits, or embedding pipeline change. Treat changed metrics as a calibration review, not an automatic default change.
