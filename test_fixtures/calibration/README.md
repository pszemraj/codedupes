# Calibration corpus

This development corpus supplies practical threshold evidence for the five languages supported by codedupes. It replaces the earlier synthetic fixtures: each project has runnable code, reviewed positive and nearby negative pairs, and search queries whose relevance is independent of pair labels. New fixtures can be small modules with behavior tests; add independent behaviors and language idioms rather than variants of one workflow just to increase the pair count.

| project | language | positives | negatives | probes |
| --- | --- | ---: | ---: | ---: |
| [ledger](ledger/README.md) | Python | 15 | 34 | 12 |
| [cowsay](../cowsay_wasm/README.md) | Rust | 13 | 29 | 12 |
| [c_metering](c_metering/README.md) | C | 19 | 32 | 12 |
| [javascript](javascript/README.md) | JavaScript | 19 | 52 | 12 |
| [harbor-ts](harbor-ts/README.md) | TypeScript | 12 | 31 | 12 |

Each language keeps at least five comparable reviewed easy pairs and five comparable reviewed medium pairs. Existing coverage is not reduced merely to balance the buckets. Easy pairs retain obvious structure, while medium pairs introduce meaningful control-flow or API differences. Hard pairs use substantial algorithmic rewrites or distributed overlap and are optional. Related-code search can provide qualitative review evidence, but it is not a corpus criterion; sparse fixtures can place a genuinely hard mate near the top of a same-language candidate pool. Keep behavior and judgments fixed when tuning thresholds. An extractable copied region is a partial positive even when its enclosing functions have different outputs. Calling a shared helper is not a second copy.

[`manifest.json`](manifest.json) is the single project and policy index. Each calibration project declares exactly one language so its measured scores map to one shipped language gate. Its annotation files use stable project-relative selectors and explicit pair judgments. Positive transitivity is never inferred. Partial pairs name checked source spans and fingerprints. Every within-family pair and every deterministic finding must have a judgment. Old annotation formats are rejected. The unit registry names judgment and probe endpoints; measurement still embeds every source candidate. Behavioral and symbol probes require at least one expected target, while only `no_result` probes may use an empty expected set. An unregistered pair above a selected gate is counted as unjudged and blocks selection readiness, so an incidental helper cannot bypass review merely because it has no registry entry.

The current corpus is development data and has no held-out split. Selection uses only projects in the development split; if evaluation projects are added, reports score them without using them to select defaults. Its size and authored challenge mix do not support claims about ecosystem-wide precision. The calibration workflow and checked result are documented in [hybrid tuning](../../docs/hybrid-tuning.md).

From the repository root, validate every contract, behavior suite, and entry point with:

```bash
conda run --name inf python scripts/validate_calibration_corpus.py --run-behavior
```
