# Harbor scheduling fixture

This small TypeScript application schedules dock loads. It keeps independent
maintenance candidates for load aggregation, dispatch planning, and booking
normalization, plus related CSV/API/webhook adapters that serve as hard
non-duplicate controls.

Run its behavior checks with:

```sh
node --experimental-strip-types --test tests/*.test.ts
node --experimental-strip-types src/main.ts
```

The fixture uses Node's built-in TypeScript type stripping so it needs no
installed JavaScript packages. The source uses only erasable TypeScript syntax.

The domain contracts are intentionally narrow: load validation happens before a
cancelled load can be ignored; accepted booking IDs are reserved only after all
fields pass validation; and dispatch holdbacks or service floors preserve
baseline output when disabled.

## Retrieval check

On 2026-09-16, `semble find-related` returned the two translated aggregation
variants at ranks 1 and 2 from `summarizeDockLoads`; the two dispatch extensions
at ranks 1 and 2 from the baseline planner; and API/webhook adapters at ranks 2
and 4 from CSV import. The adapter ranks describe related negative controls,
not duplicate candidates. The direct helper extraction also ranked first, so it
is medium. For the hard legacy
manifest pair, `admitManifestNotices` was outside the top 10 when searched from
the inline booking workflow (a type-only legacy header appeared at rank 9); the
reverse search surfaced the extracted workflow at rank 4 and the inline body at
rank 5. The behavior test proves the common admission result despite that
asymmetric retrieval shape.
