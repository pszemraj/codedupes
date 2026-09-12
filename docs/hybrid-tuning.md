# Hybrid confidence tuning

Calibrate the [per-language duplicate gates](analysis-defaults.md#semantic-duplicate-gate-defaults), [hybrid confidence split](analysis-defaults.md#hybrid-synthesis-confidence-defaults), and [search thresholds](model-profiles.md#built-in-profiles) against labeled corpora.

For one repository scan, start with the calibrated defaults and adjust the ordinary `check` scope and threshold options as needed. To change shipped gates or corroboration defaults, run these sweeps from a development checkout after installation. They load the pinned embedding models and can take time on their first run. Write exploratory reports under `scratch/`, which is ignored by Git.

## Corpora and labels

- Source of the shipped split: [polyglot calibration corpora](../test_fixtures/polyglot_calibration/README.md), swept at each language's shipped admission gate.
- Legacy guardrail: [`crab_visibility`](../test_fixtures/hybrid_tuning/crab_visibility) with [`labels.json`](../test_fixtures/hybrid_tuning/labels.json), Python clones that share identifiers and provide an optimistic bound for identifier corroboration.
- Sweep harness: [`../scripts/sweep_hybrid_gates.py`](../scripts/sweep_hybrid_gates.py)
- Semantic threshold harness: [`../scripts/sweep_semantic_thresholds.py`](../scripts/sweep_semantic_thresholds.py)

Use these synthetic corpora to check for regressions; validate changes on a real repository too.

## Recommended process

1. Run the harness on the polyglot corpora (both built-in models, every language).
2. Read the stage-1 pooled selection for the corroboration constants and the stage-2 per-language selections for the promotion gates. Inspect the `review_tp`/`review_fp` counts of each selected row: they are the labeled positives the default view gives up and the false positives it hides.
3. Re-validate on at least one real repository (`codedupes check <repo> --json --include-review` and compare `summary.duplicates_by_tier` before and after; spot-check withheld and promoted pairs by hand).
4. Keep labels/corpus changes explicit in review.

## Run the sweep

```bash
python scripts/sweep_hybrid_gates.py \
  --corpus-root test_fixtures/polyglot_calibration \
  --json-out scratch/corroboration_report.json
```

Omit `--json-out` to print results without writing a report. `--languages` and `--models` narrow the run and require at least one value; language aliases are normalized and deduplicated before selecting the corpus directory and admission gate. `--corpus-path`/`--labels-path` sweep a single legacy corpus instead of the polyglot root. Without `--corpus-root`, exactly one `--language` is required before model work begins so the sweep uses that language's admission gate. Run the bundled Python guardrail with `python scripts/sweep_hybrid_gates.py --language python`; aliases such as `py` use the same gate, and missing or repeated `--language` options are rejected.

Semantic candidates are collected once per (model, corpus) at that language's shipped admission gate (`--semantic-gate` overrides it flat), so every row scores exactly the pairs the analyzer would admit. The report records `output_policy: hybrid_high_confidence`: each row's `tp`/`fp`/`fn` and `precision`/`recall`/`f1` score the visible subset, its `published_*` fields score every published pair, and `review_tp`/`review_fp` score the withheld tier alone. The semantic sweep's same-named fields score all published pairs. Both sweeps require a model pinned to an immutable 40-character commit and embed the [calibration manifest](#semantic-threshold-sweep-model-profiles) per corpus. Fresh duplicate and hybrid reports also record the traditional Jaccard gate and tiny-pair filter that contributed candidates; older stored reports without that field must not be treated as evidence for a non-default traditional policy.

## Stages, grids, and selection

Per model the harness runs two stages:

1. **Corroboration constants**, promotion disabled, over `--weak-jaccard-grid` × `--statement-ratio-grid` (defaults `0,0.05,...,0.40` × `0,0.20,0.35,0.50,0.65,0.80`) on every corpus. Per-corpus rows are pooled by summing counts, and one global pair is selected.
2. **Promotion gate**, per corpus, at the stage-1 constants, from the admission gate up to `--high-gate-stop` (default `0.96`) in `--high-gate-step` (default `0.02`) steps plus `off`.

A row is feasible when its visible recall is at least `--recall-retention-min` (default `0.85`) of its published recall and its visible precision is at least its published precision. The pooled selection requires feasibility in every corpus, so aggregate improvements cannot conceal a language's regression. Among feasible rows the harness maximizes precision, then F1, then prefers the stricter split (higher constants, higher gate; `off` is the strictest gate). The stricter split wins ties because a gate that changes no labeled pair can still promote unvalidated pairs elsewhere. The `(0, 0)` constants and an `off` gate retain every published pair, allowing the sweep to select no filtering when the alternatives offer no benefit.

`tests/test_corroboration_reports.py` re-derives the shipped constants and gates from the recorded report and checks this policy, the same way `tests/test_calibration_reports.py` checks the admission gates.

## Semantic threshold sweep (model profiles)

Run the duplicate and search threshold sweeps for built-in model profiles:

```bash
python scripts/sweep_semantic_thresholds.py \
  --top-n 10 \
  --json-out scratch/semantic_threshold_report.json \
  --search-json-out scratch/search_threshold_report.json
```

By default this sweeps the legacy Python-only `crab_visibility` corpus; its duplicate-threshold report is a guardrail, not the source of the shipped per-language duplicate gates. Those are calibrated from the [polyglot corpus](../test_fixtures/polyglot_calibration/README.md).

To re-run one polyglot corpus, set `lang` to `c`, `rust`, `javascript`, `typescript`, or `python`:

```bash
lang=c
python scripts/sweep_semantic_thresholds.py \
  --corpus-path test_fixtures/polyglot_calibration/$lang \
  --labels-path test_fixtures/polyglot_calibration/labels/$lang.json \
  --search-probes-path test_fixtures/polyglot_calibration/search_probes/$lang.json \
  --language "$lang" \
  --duplicate-start 0.40 \
  --json-out scratch/${lang}_semantic_threshold_report.json \
  --search-json-out scratch/${lang}_search_threshold_report.json
python scripts/report_calibration_distributions.py \
  --languages "$lang" \
  --json-out scratch/${lang}_similarity_distributions.json
```

Default report paths:

- [`../test_fixtures/hybrid_tuning/semantic_threshold_report.json`](../test_fixtures/hybrid_tuning/semantic_threshold_report.json) - duplicate thresholds
- [`../test_fixtures/hybrid_tuning/search_threshold_report.json`](../test_fixtures/hybrid_tuning/search_threshold_report.json) - search thresholds, evaluated against [`../test_fixtures/hybrid_tuning/search_probes.json`](../test_fixtures/hybrid_tuning/search_probes.json)

Each report records the full calibration identity per model: the pinned immutable commit, embedding pipeline schema and runtime fingerprint, encode plan (route and prompt per input mode), the requested device plus the effective embedding-space identity the analyzer actually produced (its runtime variant covers dtype and Metal math policy, and reflects an accelerator request that fell back and restarted on CPU - thresholds are never labeled with a device or dtype that did not produce them), embedding dimension, candidate policy, candidate coverage, and SHA-256 digests of the corpus and labels/probes. Duplicate rows score the final combined output, not the raw semantic list, and each row also carries `tiers` (per-tier `predicted`/`tp`/`fp`/`precision`/`positive_share`; `precision` is `null` for an empty tier) and `visible` (full metrics of the default-visible tiers); the manifest records the `corroboration` constants and promotion gates that split was computed with, and `selected_category_recall` reports `visible_detected`/`visible_recall` per category.

Fresh duplicate reports distinguish endpoint and pair coverage. `embedded_positive_pairs` counts labels whose two units reach embedding, while `scoreable_positive_pairs` also applies the scanner's language, kind, overlap, and exact-fingerprint exclusions. Traditional matches outside that pair set are `traditional_recovered_pairs`; `reachable_positive_pairs` combines both paths, and only labels reached by neither path are `unreachable_positive_pairs`. `recall_ceiling` is this structural reachable share before cosine scores and thresholds, rather than a claim about model recall. Historical reports without `embedded_positive_pairs` used `scoreable_positive_pairs` for endpoint embedding coverage and remain unchanged.

The sweep defaults to the production statement floor and refuses to run for a model that cannot be pinned to a 40-character Hub commit - pass `--model-revision` or pin the profile's `default_revision`. Local model directories are rejected because their weights are content-fingerprinted and ignore Hub revisions, so a caller-supplied commit cannot make them immutable. Both grids are overrideable (`--duplicate-start`/`--duplicate-stop`, `--search-start`/`--search-stop`; search default 0.20-0.90), with finite bounds in the analyzer's supported `[0.0, 1.0]` range. Each grid's chosen start is also its score-collection floor, so lowering `--search-start` includes those lower-scoring candidates in every evaluated row. Every manifest records `selected_at_grid_edge`: a boundary selection is censored evidence to re-run with a wider grid, never an interior optimum.

Selection policy is deterministic and scores the complete published set (the tier split only decides default visibility):

- sort by `f1` (desc), `precision` (desc), `recall` (desc), `fp` (asc), then prefer the looser threshold on remaining ties

Transferring a swept value into a profile default is a reviewed decision, not automatic: re-validate on at least one real repository and follow the [duplicate gate selection policy](analysis-defaults.md#semantic-duplicate-gate-defaults). `tests/test_calibration_reports.py` re-derives each shipped gate from its recorded grid and checks that policy. [Model profiles](model-profiles.md#built-in-profiles) lists search defaults and their calibration evidence.
