# Hybrid confidence tuning

Tune the split between the tiers `codedupes check` shows by default and the `semantic_review` tier it withholds (see [report selection](output.md#json-schema-v3)). Admission — which semantic pairs exist at all — is set by the [per-language duplicate gates](analysis-defaults.md#semantic-duplicate-gate-defaults) and is not tuned here.

This is a maintainer workflow for changing shipped gates or corroboration defaults. It is not needed to tune one repository scan: start with the calibrated defaults, then use ordinary `check` scope and threshold options if that scan needs adjustment. Run these commands from a development checkout after installation; they load the pinned embedding models and can take time on their first run. Write exploratory reports under `scratch/`, which is ignored by Git.

## Corpora and labels

- Source of the shipped split: [`../test_fixtures/polyglot_calibration/`](../test_fixtures/polyglot_calibration/README.md) — one labeled corpus per language, swept at that language's shipped admission gate.
- Legacy guardrail: [`../test_fixtures/hybrid_tuning/crab_visibility`](../test_fixtures/hybrid_tuning/crab_visibility) with [`labels.json`](../test_fixtures/hybrid_tuning/labels.json) — Python only, identifier-sharing clones, the optimistic bound for identifier corroboration.
- Sweep harness: [`../scripts/sweep_hybrid_gates.py`](../scripts/sweep_hybrid_gates.py)
- Semantic threshold harness: [`../scripts/sweep_semantic_thresholds.py`](../scripts/sweep_semantic_thresholds.py)

Use these synthetic corpora to check for regressions; validate changes on a real repository too.

## What the split is made of

A semantic-only pair (admitted by its language gate, no exact or Jaccard evidence) is promoted to `semantic_high_confidence` when either path holds; otherwise it is `semantic_review`:

- corroboration: identifier Jaccard ≥ `hybrid_weak_identifier_jaccard_min` and statement-count ratio ≥ `hybrid_statement_ratio_min` on the model profile (`src/codedupes/semantic_profiles.py`; both signals are embedding-independent, but which candidates they must split depends on the admission gate, so the constants are swept and shipped per model — `src/codedupes/constants.py` holds the generic-profile values);
- similarity promotion: cosine ≥ the language's `language_high_confidence_thresholds` entry on the same profile (per model and language; a cross-language pair must clear the stricter gate, a language without a calibrated entry has promotion off, and an explicit `--semantic-threshold` disables this path along with the calibrated admission gates while keeping the profile's corroboration constants).

Identifier corroboration is weak for Python by construction: the Python extractor collects bound and referenced names but not attribute names, while the tree-sitter languages collect every identifier leaf, so renamed Python clones score near zero identifier overlap. The similarity path exists so that strongly similar renamed clones still reach the default view.

## Recommended process

1. Run the harness on the polyglot corpora (both built-in models, every language).
2. Read the stage-1 pooled selection for the corroboration constants and the stage-2 per-language selections for the promotion gates. Inspect the `review_tp`/`review_fp` counts of each selected row: they are the labeled positives the default view gives up and the false positives it hides.
3. Re-validate on at least one real repository (`codedupes check <repo> --json --include-review` and compare `summary.duplicates_by_tier` before and after; spot-check withheld and promoted pairs by hand).
4. Keep labels/corpus changes explicit in review.

## Run the sweep

```bash
python scripts/sweep_hybrid_gates.py \
  --corpus-root test_fixtures/polyglot_calibration \
  --json-out test_fixtures/polyglot_calibration/reports/corroboration_report.json
```

Omit `--json-out` to print results without writing a report. `--languages` and `--models` narrow the run; `--corpus-path`/`--labels-path` (with one `--language`) sweep a single legacy corpus instead of the polyglot root.

Semantic candidates are collected once per (model, corpus) at that language's shipped admission gate (`--semantic-gate` overrides it flat), so every row scores exactly the pairs the analyzer would admit. The report records `output_policy: hybrid_high_confidence`: each row's `tp`/`fp`/`fn` and `precision`/`recall`/`f1` score the visible subset (published output minus `semantic_review`), its `published_*` fields score every published pair, and `review_tp`/`review_fp` score the withheld tier alone. The semantic sweep's same-named fields score all published pairs. Both sweeps require a model pinned to an immutable 40-character commit and embed the [calibration manifest](#semantic-threshold-sweep-model-profiles) per corpus.

## Stages, grids, and selection

Per model the harness runs two stages:

1. **Corroboration constants**, promotion disabled, over `--weak-jaccard-grid` × `--statement-ratio-grid` (defaults `0,0.05,…,0.40` × `0,0.20,0.35,0.50,0.65,0.80`) on every corpus. Per-corpus rows are pooled by summing counts, and one global pair is selected.
2. **Promotion gate**, per corpus, at the stage-1 constants, from the admission gate up to `--high-gate-stop` (default `0.96`) in `--high-gate-step` (default `0.02`) steps plus `off`.

A row is feasible when its visible recall is at least `--recall-retention-min` (default `0.85`) of its published recall **and** its visible precision is not below its published precision — hiding review pairs must buy precision and may cost bounded recall. The pooled selection additionally requires feasibility in every corpus, so no single language's default view can be gutted by a constant that helps the others. Among feasible rows the harness maximizes precision, then F1, then prefers the stricter split (higher constants, higher gate; `off` is the strictest gate). Stricter wins ties because a gate that buys nothing on the corpus still promotes pairs off-corpus — the first polyglot sweep's looser tie-break picked a Python gate that changed no corpus row and promoted 88 mixed-quality pairs on this repository. The `(0, 0)` constants and an `off` gate equal all-published output, so "the split does nothing useful" is a detectable outcome rather than a hidden one.

`tests/test_corroboration_reports.py` re-derives the shipped constants and gates from the recorded report and checks this policy, the same way `tests/test_calibration_reports.py` checks the admission gates.

## Semantic threshold sweep (model profiles)

Run the duplicate and search threshold sweeps for built-in model profiles:

```bash
python scripts/sweep_semantic_thresholds.py \
  --top-n 10 \
  --json-out scratch/semantic_threshold_report.json \
  --search-json-out scratch/search_threshold_report.json
```

By default this sweeps the legacy Python-only `crab_visibility` corpus; its duplicate-threshold report is a guardrail, not the source of the shipped per-language duplicate gates. Those are calibrated from [`../test_fixtures/polyglot_calibration/`](../test_fixtures/polyglot_calibration/README.md), whose README records the per-language re-run command (`--corpus-path`, `--labels-path`, `--language`, `--skip-search`, and `--duplicate-start`/`--duplicate-stop` to widen the grid below the default floor).

Default report paths:

- [`../test_fixtures/hybrid_tuning/semantic_threshold_report.json`](../test_fixtures/hybrid_tuning/semantic_threshold_report.json) - duplicate thresholds
- [`../test_fixtures/hybrid_tuning/search_threshold_report.json`](../test_fixtures/hybrid_tuning/search_threshold_report.json) - search thresholds, evaluated against [`../test_fixtures/hybrid_tuning/search_probes.json`](../test_fixtures/hybrid_tuning/search_probes.json)

Each report records the full calibration identity per model: the pinned immutable commit, embedding pipeline schema and runtime fingerprint, encode plan (route and prompt per input mode), the requested device plus the effective embedding-space identity the analyzer actually produced (its runtime variant covers dtype and Metal math policy, and reflects an accelerator request that fell back and restarted on CPU - thresholds are never labeled with a device or dtype that did not produce them), embedding dimension, candidate policy, candidate coverage, and SHA-256 digests of the corpus and labels/probes. Duplicate rows score the final combined output, not the raw semantic list, and each row also carries `tiers` (per-tier `predicted`/`tp`/`fp`/`precision`/`positive_share`; `precision` is `null` for an empty tier) and `visible` (full metrics of the default-visible tiers); the manifest records the `corroboration` constants and promotion gates that split was computed with, and `selected_category_recall` reports `visible_detected`/`visible_recall` per category. Labels outside the semantic candidate policy can still be recovered by full-scope traditional analysis; the manifest records those recoveries separately and counts only labels reached by neither tier as unreachable. The sweep defaults to the production statement floor and refuses to run for a model that cannot be pinned to a 40-character commit - pass `--model-revision` or pin the profile's `default_revision`. Both grids are overrideable (`--duplicate-start`/`--duplicate-stop`, `--search-start`/`--search-stop`; search default 0.20-0.90), and every manifest records `selected_at_grid_edge`: a boundary selection is censored evidence to re-run with a wider grid, never an interior optimum.

Selection policy is deterministic and scores the complete published set (the tier split only decides default visibility):

- sort by `f1` (desc), `precision` (desc), `recall` (desc), `fp` (asc), then prefer the looser threshold on remaining ties

Transferring a swept value into a profile default is a reviewed decision, not automatic: re-validate on at least one real repository and follow the [duplicate gate selection policy](analysis-defaults.md#semantic-duplicate-gate-defaults). `tests/test_calibration_reports.py` re-derives each shipped gate from its recorded grid and checks that policy. [Model profiles](model-profiles.md#built-in-profiles) lists search defaults and their calibration evidence.
