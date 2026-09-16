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
