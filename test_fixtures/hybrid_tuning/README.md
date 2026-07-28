# Hybrid gate tuning fixtures

This directory contains the synthetic corpus and labels used by the
[hybrid gate tuning workflow](../../docs/hybrid-tuning.md).

## Contents

- [`crab_visibility`](crab_visibility): deterministic Python corpus
- [`labels.json`](labels.json): expected duplicate groups

## Constraints

- Do not tune thresholds from this corpus alone; validate against at least one real repository.
- Keep corpus deterministic and free from generated artifacts.
