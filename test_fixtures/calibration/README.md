# Calibration corpus

This development corpus supplies practical threshold evidence for the five
languages supported by codedupes. It replaces the earlier synthetic fixtures.
Each project is a runnable application with reviewed positive and nearby
negative pairs plus search queries whose relevance is independent of the pair
labels.

| project | language | positives | negatives | probes |
| --- | --- | ---: | ---: | ---: |
| ledger | Python | 13 | 30 | 12 |
| cowsay | Rust | 8 | 29 | 12 |
| c_metering | C | 20 | 31 | 12 |
| javascript | JavaScript | 17 | 53 | 12 |
| harbor-ts | TypeScript | 11 | 32 | 12 |

Positive pairs cover easy, medium, and hard cases. Easy pairs retain obvious
structure, medium pairs introduce meaningful control-flow or API differences,
and hard pairs use rewrites or partial overlap that local Semble checks place
well down the result list or omit in the annotated direction. Difficulty is a
fixture descriptor; model scores never change the labels.

[`manifest.json`](manifest.json) is the single project and policy index. Its
annotation files use stable project-relative selectors and explicit pair
judgments. Positive transitivity is never inferred. Partial pairs name checked
source spans and fingerprints. Every within-family pair and every deterministic
finding must have a judgment. Old annotation formats are rejected.

All projects and related variants are development data. The corpus is intended
to set useful defaults, while its size and authored challenge mix do not support
claims about ecosystem-wide precision. Raw CPU and MPS measurements are local
scratch artifacts; the compact checked result records the selected settings and
device comparison.

From the repository root, validate every contract, behavior suite, and entry
point with:

```bash
conda run --name inf python scripts/validate_calibration_corpus.py --run-behavior
```

See [the calibration workflow](../../docs/hybrid-tuning.md) for measurement and
selection commands.
