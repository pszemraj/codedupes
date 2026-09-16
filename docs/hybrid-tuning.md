# Calibration measurements and replay

The issue #20 pilot replaces the previous calibration sources and report
formats. Its authoritative [manifest and annotations](../test_fixtures/calibration/README.md)
cover one Python ledger application and the runnable Rust cowsay application.
Both projects are development data. The pilot measures the two built-in model
profiles and exercises production decisions, but it does not select replacement
thresholds or corroboration constants.

The values shipped in the model profiles remain unchanged and are recorded in
[`frozen_defaults.json`](../test_fixtures/calibration/frozen_defaults.json). The
snapshot is a change guard. It is not evidence that the values are optimal.

## Validate the contract and behavior

Run commands through the project environment:

```bash
conda run --name inf python scripts/validate_calibration_corpus.py \
  --run-behavior
```

The validator resolves stable unit selectors, verifies partial-region hashes,
requires every within-family pair judgment, checks independent search relevance,
and runs each project's declared tests and entry point. Named policies inherit
the default policy and change one selection dimension at a time:

```bash
conda run --name inf python scripts/validate_calibration_corpus.py --policy tests
conda run --name inf python scripts/validate_calibration_corpus.py --policy public
conda run --name inf python scripts/validate_calibration_corpus.py --policy classes
```

Version 2 is the only accepted annotation format. The scripts deliberately have
no legacy corpus-root or label-path compatibility mode.

## Capture reusable measurements

Each measurement process accepts exactly one project, model, and device. CPU is
the reproducible fp32 reference. MPS is an independent device comparison and
must run in a fresh process on the real accelerator. Embedding caches are
disabled in both cases.

```bash
conda run --name inf python scripts/measure_calibration.py \
  --project ledger --model gte-modernbert-base --device cpu
conda run --name inf python scripts/measure_calibration.py \
  --project ledger --model gte-modernbert-base --device mps
```

Repeat those commands for `cowsay` and `embeddinggemma-300m`. Artifacts live in
`test_fixtures/calibration/measurements/<project>/<policy>/<model>/<device>/`.
Each directory contains metadata plus JSON Lines tables for units, unordered
unit pairs, and query-unit scores.

The pair and query tables store scores before threshold filtering. Unit records
include statement counts, identifiers, source hashes, endpoint eligibility, and
explicit missing-score reasons. Pair records add deterministic relationships,
identifier overlap, statement ratios, and pair-comparability exclusions. The
metadata records pinned model revisions, duplicate and search task identities,
prompts/document preparation, effective devices, runtime variants, and timings.

Source, extraction, embedding-policy, model, or query changes invalidate the
affected tables. Label and threshold changes replay existing scores. Changes to
tests or other evidence invalidate behavior evidence without forcing unchanged
source to be embedded again.

## Replay production policy

The report command evaluates the shipped production settings and compares CPU
with MPS:

```bash
conda run --name inf python scripts/report_calibration_distributions.py \
  --json-out test_fixtures/calibration/reports/pilot-summary.json
```

Reports separate complete pipeline output, the default-visible subset,
deterministic recovery, semantic contribution beyond deterministic findings,
and semantic performance conditional on pair eligibility. Precision counts true
positives only from reviewed positive judgments and false positives only from
reviewed negatives. Reviewed ambiguities and unjudged findings remain explicit,
and reports call the corresponding precision `judged_only_precision`.

Search reporting includes threshold-level metrics, precision and recall at the
production top-10 limit, and no-result-query behavior. Device comparison records
score drift and any default-threshold decision differences.

The review queue is the union of authored family pairs, deterministic findings,
both models' published findings at shipped settings, and a seed-20 sample of up
to 20 remaining background pairs per project. An unresolved row makes selection
ineligible. Broader exploratory sweep findings go back into that queue.

## Inspect threshold and hybrid grids

The semantic/search sweep replays raw scores instead of embedding again:

```bash
conda run --name inf python scripts/sweep_semantic_thresholds.py \
  --json-out scratch/pilot-thresholds.json
```

The hybrid sweep replays the production synthesis over identifier-overlap and
statement-ratio grids:

```bash
conda run --name inf python scripts/sweep_hybrid_gates.py \
  --json-out scratch/pilot-hybrid.json
```

Both outputs have `selection: null`. Adjacent settings with identical decisions
are represented as indifference plateaus; the pilot does not promote one of
them. A future recalibration requires all five supported languages, independently
authored applications, an untouched group-disjoint evaluation split, uncertainty
estimates, and real-repository transfer checks before changing defaults.
