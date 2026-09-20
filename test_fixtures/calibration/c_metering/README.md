# C metering calibration fixture

This is a small C11 meter-reconciliation application. It keeps intentionally independent implementations of five real maintenance shapes:

- three aggregation algorithms that agree on successful aggregations after validating each reading before discard filtering, while retaining their distinct failure precedence;
- a baseline payout planner and two minimum-payout extensions;
- inline, extracted-parser, and two-phase row import workflows;
- CSV, API, and idempotent-delivery adapters that share strict normalization but differ in file ownership, reporting, and delivery acceptance.
- bounded byte-span record parsers that use a state scan or field slicing and standard conversion while preserving caller output on failure.

Labels and pair judgments follow the shared [calibration corpus contract](../README.md).

The project uses value structs, pointer-based output parameters, and a public header so extraction sees ordinary C application boundaries. `make test` runs the behavioral/differential suite and `make run` executes the demo.

Aggregate and deferred-payout totals return `METERING_OVERFLOW` rather than overflowing `int`; payout planners also reject device fields without an in-array NUL terminator. The aggregation variants remain positive duplicate judgments, but the sort-first implementation validates all readings and aggregates in key order, while the streaming variants can report output capacity first. Their behavioral-equivalence evidence therefore covers only successful calls.
