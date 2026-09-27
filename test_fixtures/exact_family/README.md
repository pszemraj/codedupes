# Exact family fixture

A small Python tree whose only findings are exact duplicates, planted to exercise how reports group pairwise exact edges into families instead of listing `n(n-1)/2` edges. It is not part of the calibration corpus and carries no pair annotations.

## Planted families

| Family | Members | Fingerprint | What differs |
| --- | --- | --- | --- |
| `render_receipt` | `invoices.py`, `receipts.py`, `refunds.py`, `statements.py`, `exports.py` | `token_hash` | Nothing: five token-for-token copies (ten pairwise edges, one family). |
| `sum_amounts` / `sum_credits` | `totals.py` | `structural_hash` | Every binding name and the string literals; the structure is identical. Attribute names are preserved by the structural fingerprint as API shape, so the pair reads its records through string-keyed subscripts. |

`headers.py` holds one unrelated function so the tree has a unit outside both families. Numeric literals are not normalized by the structural fingerprint, so the renamed pair keeps `0` in both bodies on purpose.

## Analyze the fixture

From the repository root:

```bash
codedupes check test_fixtures/exact_family --traditional-only --no-unused
```

The report lists two families and no pairwise exact edges; `--json` puts them under `exact_families` with `duplicates_by_tier.exact == 2` and `exact_family_members == 7`.
