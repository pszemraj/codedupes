# C metering calibration fixture

This is a small C11 meter-reconciliation application. It keeps intentionally
independent implementations of four real maintenance shapes:

- three aggregation algorithms that validate every reading before discarding it;
- a baseline payout planner and two minimum-payout extensions;
- inline, extracted-parser, and two-phase row import workflows;
- CSV, API, and idempotent-delivery adapters that share strict normalization but
  differ in file ownership, reporting, and delivery acceptance.
- bounded byte-span record parsers that use a state scan or field slicing and
  standard conversion while preserving caller output on failure.

The project uses value structs, pointer-based output parameters, and a public
header so extraction sees ordinary C application boundaries. `make test` runs
the behavioral/differential suite and `make run` executes the demo.
