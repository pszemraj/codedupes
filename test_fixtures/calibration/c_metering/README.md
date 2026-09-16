# C metering calibration fixture

This is a small C11 meter-reconciliation application. It keeps intentionally
independent implementations of four real maintenance shapes:

- three aggregation algorithms that validate every reading before discarding it;
- a baseline payout planner and two minimum-payout extensions;
- inline, extracted-parser, and two-phase row import workflows;
- CSV, API, and idempotent-delivery adapters that share strict normalization but
  differ in file ownership, reporting, and delivery acceptance.

The project uses value structs, pointer-based output parameters, and a public
header so extraction sees ordinary C application boundaries. `make test` runs
the behavioral/differential suite and `make run` executes the demo.

## Retrieval observation

Semble `find-related` checks (seedless local index, top 10) found the easy
aggregation variants at ranks 1 and 2, the medium payout policy form at rank
1 and baseline at rank 5, and the refactored row workflow at rank 1. The hard
idempotent endpoint found the regular API adapter at rank 5; the CSV endpoint
was absent from its top 10. These are observations, not fixture targets: the
row pair remains hard because staging versus immediate emission changes its
implementation shape, while its shared import vocabulary still retrieves well.
