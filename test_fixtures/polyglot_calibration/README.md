# Polyglot calibration corpora

Synthetic per-language duplicate-detection corpora for C, Rust, JavaScript, and TypeScript, with labeled clone pairs. They sanity-check that the Python-calibrated defaults behave sensibly on the other supported languages. They are smoke corpora, not benchmarks: no per-language default threshold has been derived from them.

## Layout

- `c/`, `rust/`, `javascript/`, `typescript/`: ~6 files per language, 28–31 extracted units each, one coherent domain per language.
- `labels/<language>.json`: `positive_groups` of `filename::name` specs (mutual clones), plus informational `negative_controls` (hard same-domain non-clones the analyzer must not report as exact).

Each corpus contains, as labeled positives: 3 exact-copy pairs, 3 alpha-renamed pairs (structural-hash equal), 3 reformat-only pairs (token-hash equal), and 6–8 near-clone pairs (renamed plus one real behavioral change). Near clones are deliberately below the identifier-Jaccard range so they exercise the semantic tier.

## Authoring constraints

- Every unit name is unique within its file (`sweep_common.resolve_label_unit` requires a unique `filename::name` match).
- No filename may match the default test-exclusion globs (`test_*`, `*_test.*`, `*.test.*`, `*.spec.*`).
- The C corpus must stay free of C++ extensions or the `.h` ambiguity policy changes.
- No Rust `#[cfg(test)]`/`#[test]` code: the extractor skips it.
- Bodies are 3–8 statements so units pass the default semantic gates.

## Recorded results (2026-08-14, gte-modernbert-base pinned profile, default thresholds, `min_semantic_statements=0`)

Deterministic tiers transfer cleanly:

- All 9 exact/renamed/reformat pairs per language are reported as `exact`, with zero false positives (precision 1.0) and no negative control at any exact tier.
- The whole-tree run (119 units, all four languages plus no filter) reports zero cross-language pairs.
- Zero parse diagnostics across all corpora.
- One known exception: the JavaScript CLASS pair (`RetryBudget`/`AttemptBudget`) is missing from combined-mode output because combined mode currently narrows traditional analysis to semantic candidates (functions/methods). Traditional-only mode reports all 9/9. This is the documented analysis-stage scope defect scheduled for its own correction PR; this corpus is its regression demonstration.

The semantic layer does not transfer at the Python-calibrated duplicate threshold (0.96):

| language | near-clone cosine range (median) | hard-negative cosine range | near clones ≥ 0.96 |
| --- | --- | --- | --- |
| python (crab_visibility, for scale) | 0.83–0.99 (0.97) | — | 4/7 |
| c | 0.69–0.76 (0.74) | 0.56–0.67 | 0/6 |
| rust | 0.65–0.77 (0.73) | 0.59–0.67 | 0/6 |
| javascript | 0.57–0.85 (0.73) | 0.49–0.50 | 0/6 |
| typescript | 0.68–0.85 (0.73) | 0.52–0.63 | 0/7 |

Interpretation: the embedding similarity scale for non-Python code sits far below Python's, so semantic near-clone recall at default gates is 0 for all four languages. Near-clone and hard-negative distributions overlap at the edges (worst Rust near clone 0.65 vs best Rust negative 0.67), so usable per-language thresholds exist in the ~0.7–0.8 region but must come from a real calibration corpus with many more labeled decisions, not from this smoke set. Note the difficulty is not identical across corpora: these near clones are fully alpha-renamed, while the Python corpus's near pairs share identifier fragments.

## Re-running

```bash
conda run -n inf python scripts/sweep_hybrid_gates.py --corpus-path test_fixtures/polyglot_calibration/rust --labels-path test_fixtures/polyglot_calibration/labels/rust.json --language rust
```

Repeat per language. Keep corpus and label changes explicit in review; if a grammar pin bump changes any recorded number above, understand why before accepting it.
