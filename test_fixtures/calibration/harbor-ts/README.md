# Harbor scheduling fixture

This small TypeScript application schedules dock loads. It keeps independent
maintenance candidates for load aggregation, dispatch planning, booking
normalization, and typed registration validation, plus related CSV/API/webhook
adapters that serve as hard non-duplicate controls.

Run its behavior checks with:

```sh
node --experimental-strip-types --test tests/*.test.ts
node --experimental-strip-types src/main.ts
```

The fixture uses Node's built-in TypeScript type stripping so it needs no
installed JavaScript packages. The source uses only erasable TypeScript syntax.

The domain contracts are intentionally narrow: load validation happens before a
cancelled load can be ignored; booking weights, load totals, and dispatch
thresholds must fit JavaScript's safe-integer range; accepted booking IDs are
reserved only after all fields pass validation; and dispatch holdbacks or
service floors preserve baseline output when disabled while rejecting unsafe
held totals. Registration validation preserves ordered field issues across a narrowed branch
implementation and an independent typed rule table for person and business
submissions.
