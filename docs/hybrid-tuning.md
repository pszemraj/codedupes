# Calibrating semantic and hybrid defaults

The first implementation phase for Issue #20 replaces the old synthetic inputs
with five runnable development applications. The
[corpus contract](../test_fixtures/calibration/README.md)
contains reviewed easy, medium, and hard duplicate pairs plus independent search
relevance for Python, C, Rust, JavaScript, and TypeScript.

## Checked development-corpus result

The checked [calibration result](../test_fixtures/calibration/calibration-results.json)
contains 118 explicitly annotated units, 78 positive judgments, 178 reviewed
negative judgments, and 60 search probes. Of the positives, 77 are comparable
semantic pairs; one exact Rust pair is retained as a deterministic fixture but
excluded from semantic calibration. Every language has at least five comparable
easy pairs and five comparable medium pairs. This is authored development data,
not a held-out estimate of accuracy in arbitrary repositories.

This result does not complete Issue #20. The remaining phase is a compact,
hand-reviewed real-code check in every supported language, followed by only the
targeted corpus additions and recalibration that those checks justify. Larger
quotas, ecosystem-level statistical claims, cross-language calibration, and
unrelated analyzer work are outside the issue unless that evidence exposes a
specific need. Until the real-code checks pass, these settings are development
defaults rather than evidence that arbitrary repositories will have clean top
results.

Duplicate recall at the selected per-language admission gates is shown as
detected / labeled comparable positive pairs:

| language | GTE easy | GTE medium | GTE hard | Gemma easy | Gemma medium | Gemma hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python | 3/5 | 3/10 | — | 4/5 | 8/10 | — |
| Rust | 5/5 | 5/6 | 0/1 | 5/5 | 5/6 | 0/1 |
| C | 4/5 | 8/14 | — | 5/5 | 13/14 | — |
| JavaScript | 5/5 | 8/14 | — | 4/5 | 8/14 | — |
| TypeScript | 5/5 | 4/6 | 0/1 | 5/5 | 4/6 | 0/1 |
| **overall** | **22/25** | **28/50** | **0/2** | **23/25** | **38/50** | **0/2** |
| **recall** | **88.0%** | **56.0%** | **0.0%** | **92.0%** | **76.0%** | **0.0%** |

A dash means that language has no labeled hard pair. Difficulty belongs to
positive examples, so false positives and therefore precision and F1 do not have
a difficulty bucket. The pooled metrics below use every comparable reviewed pair.
Default-visible duplicates apply the jointly selected hybrid visibility policy
after semantic admission.

| output | model | TP / FP / FN | precision | recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| duplicate admission | GTE | 51 / 6 / 27 | 89.5% | 65.4% | 75.6% |
| duplicate admission | Gemma | 62 / 12 / 16 | 83.8% | 79.5% | 81.6% |
| default-visible duplicates | GTE | 51 / 1 / 27 | 98.1% | 65.4% | 78.5% |
| default-visible duplicates | Gemma | 59 / 2 / 19 | 96.7% | 75.6% | 84.9% |
| search | GTE | 72 / 20 / 22 | 78.3% | 76.6% | 77.4% |
| search | Gemma | 62 / 18 / 32 | 77.5% | 66.0% | 71.3% |

Both search profiles keep all 10 no-result probes clean. Selection discards
settings below 50% judged precision, maximizes F1, and prefers recall within
`0.005` F1 of the optimum. Duplicate and search plateaus use their stable midpoint. Hybrid
visibility jointly selects corroboration and per-language promotion gates after
the admission gates are fixed. CPU fp32 supplies the reference scores; independent
uncached MPS fp32 runs produce no duplicate, visibility-tier, or search decision
changes. Maximum observed pair or query score drift is `7.45e-7`.

## Real-repository smoke check

As the first phase-two check, both profiles were run uncached on live MPS over
this repository's production and maintenance Python (`src/` and `scripts/`).
Tests, calibration fixtures, local scratch data, generated distributions, and
tool metadata were excluded. The scan found no exact, structurally similar,
traditional-near, or hybrid-confirmed pair. Manual review of the leading
semantic-only results found wrappers, caller/callee pairs, sibling operations,
and parallel parser backends rather than code that should be consolidated.

| profile | merge-base semantic admissions | phase-one semantic admissions | merge-base default-visible | phase-one default-visible |
| --- | ---: | ---: | ---: | ---: |
| GTE | 529 | 83 | 139 | 64 |
| Gemma | 245 | 245 | 218 | 145 |

The phase-one policy materially reduces noise, especially for GTE, but does not
meet Issue #20's real-code precision bar. Similarity-only threshold increases do
not cleanly separate the reviewed fixture positives from these related-but-
distinct functions. Treating every semantic candidate as review-only removes
useful recall as well. The follow-up should add a compact representative negative
slice, then recalibrate and repeat the same check across all five languages;
this report deliberately does not hide the failure with a repository-specific
filter or an unmeasured threshold change.

Validate source selectors, pair coverage, tests, and entry points first:

```bash
conda run --name inf python scripts/validate_calibration_corpus.py --run-behavior
```

Measure one project, model, and device in each fresh process. Measurements write
to ignored `scratch/calibration/` files and disable embedding-cache reuse.

```bash
conda run --name inf python scripts/measure_calibration.py \
  --project ledger --model gte-modernbert-base --device cpu
conda run --name inf python scripts/measure_calibration.py \
  --project ledger --model gte-modernbert-base --device mps
```

Repeat for `ledger`, `cowsay`, `c_metering`, `javascript`, and `harbor-ts`, and
for `gte-modernbert-base` and `embeddinggemma-300m`. CPU fp32 is the reference;
MPS is an independent real-device comparison. Raw artifacts record the Python,
PyTorch, Transformers, and Sentence Transformers versions and include them in
their measurement identity, along with the inference batch size. Derived
selections also record digests of the complete CPU score matrices, while the
checked report records every CPU and MPS measurement digest and validates its
runtime, dtype, batch, and fresh-execution provenance. Replacing or truncating a
raw artifact therefore forces regeneration. Before a raw artifact is used, the
loader also recomputes semantic-candidate inclusion, pair eligibility and
exclusion, traditional evidence, identifier overlap, and statement ratio from
the corpus; serialized routing fields cannot alter a sweep. A source,
annotation unit ID or selector, model, extraction, task, prompt, runtime, or
dtype-policy change makes an artifact stale.
Pair judgments, search relevance, and threshold edits reuse its raw scores;
reordering annotation units also preserves measurement identity.
Source identity follows production extraction and explicitly annotated source
files; generated bytecode, build debris, and Finder metadata are not inputs.
Declared support files are validated and fingerprinted in the derived-selection
context: changing behavior evidence requires reselection, but not new embeddings.

Keeping CPU as the calibration reference is a reproducibility convention, not a
separate runtime threshold policy. The checked development-corpus comparison found a
maximum CPU/MPS score drift of `7.45e-7` and no duplicate or search decision
changes for either built-in model. Treat CPU fp32 and MPS fp32 as functionally
equivalent for these profiles, and use the default `device=auto` during normal
macOS development so Apple silicon selects the faster MPS path. Continue to run
the independent CPU/MPS comparison when source, model, extraction, task, prompt,
or dtype policy changes.

Select per-language duplicate gates and a candidate global top-10 search gate
from the CPU measurements. The sweep and report `--models` options accept
built-in profile keys, canonical model names, or recognized aliases:

```bash
conda run --name inf python scripts/sweep_semantic_thresholds.py \
  --json-out scratch/calibration/threshold-selection.json
```

Custom sweep ranges include both endpoints, even when the step does not land
on the stop value. The result records the candidate grids and an 11-point search
window centered on each selected floor; shipped-setting metrics are evaluated
at the exact shipped thresholds independently of that grid.

If the output reports unjudged or ambiguous predictions, review those source
pairs and rerun the sweep. Derived selections record their source, model, policy,
project scope, annotation, selection-code, and profile-policy identities. Label,
relevance, selection-code, or profile-default corrections reuse
raw scores but require new selections; the hybrid sweep and report reject stale
inputs, recompute every stored decision from the bound score matrices, check
which admission selection the hybrid sweep used, and require the reported
selections to equal the shipped model-profile defaults.
Once every admission selection is ready, select the
hybrid visibility constants and per-language promotion gates:

```bash
conda run --name inf python scripts/sweep_hybrid_gates.py \
  --threshold-selection scratch/calibration/threshold-selection.json \
  --json-out scratch/calibration/hybrid-selection.json
```

Search, duplicate admission, and hybrid visibility share one policy: discard
settings below 50% judged precision, maximize judged F1, then prefer higher
recall among settings within `0.005` F1 of the optimum. Fewer unresolved
predictions take precedence, followed by recall and precision. Hybrid selection
requires the precision floor in every language and in the pooled result, then
jointly searches the corroboration constants and per-language promotion gates;
promotion candidates span each admission gate through `1.0`. It does not lock
corroboration before considering promotion. Duplicate/search
threshold ties use a stable midpoint; hybrid ties prefer lower corroboration
floors, then disabled promotion, then lower promotion gates. Reports
keep reviewed ambiguities and unjudged findings separate from judged-only
precision, and break positive recall out by authored difficulty.
Search selection additionally requires every explicit no-result probe to stay
empty. Duplicate admission excludes pairs already recovered by a traditional
method because their publication does not depend on the semantic gate.

Run the multi-domain search smoke test against the selected default:

```bash
CODEDUPES_SMOKE_SEARCH=1 conda run --name inf pytest tests/test_semantic_smoke.py -k search
```

Keep these queries unchanged. Check that each target ranks in the top three
without a score floor, that emitted default hits are relevant, and that no-result
queries stay empty. The smoke pins six surfaced targets for GTE and two for Gemma;
Gemma's other four targets remain correctly ranked first but below its floor.
Do not lower the default merely to return every known target: precision and recall
are jointly evaluated by the corpus F1 policy. The shipped search floors are
`0.68` for GTE and `0.55` for Gemma. The result records the selected settings,
nearby search curve, and current shipped metrics.

After applying accepted profile values, rerun both selection commands above so
the recorded current metrics and policy identity match the shipped defaults.
Capture fresh MPS results, then write the compact checked summary:

```bash
conda run --name inf python scripts/report_calibration_distributions.py \
  --json-out test_fixtures/calibration/calibration-results.json
```

The summary records timings, effective devices, replay parity, score drift, and
duplicate/search decision changes. Duplicate comparisons include tier changes,
so a promotion or default-visibility difference cannot pass as an unchanged pair.
Its `measurement_runtime` summary derives
the PyTorch version and device scope from the included reports; mixed PyTorch
versions are rejected. Raw score matrices and model caches stay out of Git.
