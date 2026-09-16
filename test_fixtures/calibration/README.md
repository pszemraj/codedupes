# Calibration development pilot (issue #20)

This replaces the earlier synthetic polyglot calibration corpus. It provides
executable Python ledger and Rust cowsay applications, explicit maintenance
judgments, independent search relevance, and measurements for both pinned
embedding models. Everything here is **development data**. It does not justify
new production thresholds, nor demonstrate language-wide or held-out accuracy.
Shipped defaults are frozen in `frozen_defaults.json`; that snapshot is a change
guard, not evidence that those values are optimal.

The ledger aggregation source and its initial tests were adapted from the
user-supplied September 16, 2026 issue-20 authoring kit. The remaining ledger
workflows implement that kit's PY-2, PY-3, and PY-4 contracts. Cowsay source is
referenced in place. All sources and related variants remain in development;
none is an untouched evaluation sample. The application README documents the
adopted domain contracts before model scoring.

## Authoritative contract

`manifest.json` selects projects and named policies. All calibration commands
accept `--manifest`, repeatable `--project`, and `--policy`. Version 2 is the only
supported format. Project roots are relative to the manifest; selectors and
evidence files are relative to their project. Analysis roots, test roots,
support files, command argument vectors, languages, and split groups are explicit.
Support files and valid files yielding zero units are not extraction errors.

Each annotations file contains `units`, `families`, `pairs`, and `probes`:

- Units have stable `id` values and a structured `selector` with `path`,
  `qualified_name`, and `kind`. Repeated definitions require `start_line`.
  A selector must resolve exactly once; byte-position runtime UIDs are not labels.
- Pair `judgment` is `positive` (useful consolidation), `negative` (reviewed lack
  of substantial consolidation), or `ambiguous` (reviewed unresolved judgment).
  No pair record means unjudged, not negative. Every pair has tags, a contract,
  equivalence domain, maintenance rationale, and executable evidence references.
  `behavior_equivalent` is independent of maintenance judgment. False claims
  require a concrete `difference_witness`. A true claim applies only within its
  named domain, not every possible caller configuration.
- A `partial` pair records both enclosing units and absolute one-based inclusive
  region spans with exact source SHA-256 fingerprints. Stale regions fail closed.
  A detector receives credit for the enclosing pair, not region localization.
- Every within-family combination needs a judgment. Positive transitivity is
  never inferred. Deterministic false positives are legal ground truth.
- Optional `expected_eligibility` and broad signal bands are extraction-policy
  regression contracts. Actual counts, scores, routes, and runtime status belong
  in generated measurements, not handwritten labels.
- Probes independently specify query relevance over the entire declared project
  search population (`relevance_complete: true`). Empty expected sets express
  no-result queries. Co-relevance does not imply duplicate maintenance work.

Related variants, copied helpers, and ports must share a `split_group`; source
trees cannot span splits. An eventual evaluation set must be independently
authored and assigned before inspecting scores. Family and pair counts are
coverage descriptors, not independent sample counts.

## Measurement and interpretation

Run commands from the repository root using `conda run --name inf`. The validator
can execute declared behavior commands with `--run-behavior`. Measurement runs
use one device per process; generate CPU and real MPS artifacts independently
with embedding-cache reuse disabled. The two production tasks, semantic
similarity and code retrieval, have separate identities and score tables.

Raw tables contain scores before any operating threshold, along with explicit
endpoint and comparable-pair exclusions. Evaluation reuses those tables and
production hybrid synthesis. Annotation-only edits do not require re-embedding;
source, pipeline, model, and embedding-policy changes invalidate measurements.
Tests/support have a separate execution-evidence fingerprint.

True/false positives use reviewed positive/negative judgments only. Reports name
judged-only precision, unresolved output, and its precision bounds separately.
Default-visible and complete published output, deterministic recovery,
incremental semantic contribution, and eligibility-conditional recall are
different populations. Unresolved rows cannot support selection. This pilot
selects no replacement settings, even for fully reviewed rows. Expanded sweeps
produce annotation queues; equal-decision settings are marked indifferent.

Review includes authored families, deterministic findings, both models' shipped
output, and up to 20 reproducibly sampled remaining pairs per project (seed 20).
That deliberately enriched challenge population does not estimate deployment
precision. Later work requires more independent applications, all five
languages, group-disjoint evaluation, uncertainty estimates, and real-repository
transfer checks. A positive identifier floor is not an acceptance criterion.
