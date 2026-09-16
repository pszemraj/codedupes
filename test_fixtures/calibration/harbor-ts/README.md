# Harbor scheduling fixture

This small TypeScript application schedules dock loads. It deliberately keeps
independent implementations of four maintenance candidates: load aggregation,
dispatch planning, booking normalization, and CSV/API/webhook ingestion.

Run its behavior checks with:

```sh
node --experimental-strip-types --test tests/*.test.ts
node --experimental-strip-types src/main.ts
```

The fixture uses Node's built-in TypeScript type stripping so it needs no
installed JavaScript packages. The source uses only erasable TypeScript syntax.

The domain contracts are intentionally narrow: load validation happens before a
cancelled load can be ignored; accepted booking IDs are reserved only after all
fields pass validation; and dispatch holdbacks preserve baseline output when
their threshold is zero.

## Retrieval check

On 2026-09-16, `semble find-related` returned the two translated aggregation
variants at ranks 1 and 2 from `summarizeDockLoads`; the two dispatch extensions
at ranks 1 and 2 from the baseline planner; the extracted booking workflow at
rank 1 and its scalar normalizer at rank 5 from the inline workflow; and the API
and webhook ingress variants at ranks 2 and 4 from CSV import. These observations
are recorded as retrieval shape only: easy, medium, and hard labels describe
maintenance scope and exercised behavior, not whether a local search ranks them.
