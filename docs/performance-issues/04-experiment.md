Spec for [GitHub issue #25](https://github.com/agmxyz/tiktag/issues/25), child of
[#21](https://github.com/agmxyz/tiktag/issues/21).

## Start gate

Natively blocked by #24. Do not assign or implement this task until the user has
reviewed the validated baseline and explicitly accepted exactly one falsifiable
experiment proposal, its tradeoffs, and its scope. Root records that review in
this issue before unblocking work. No candidate is selected yet.

## Reviewed resolution

After #24, the user reviewed the single tokenizer-reuse proposal and chose a
no-runtime-change conclusion. #25 is closed as not planned. No candidate was
assigned or implemented; no comparison or reversed-order confirmation artifacts
exist, and none of the candidate retention gates were exercised. The proposed
window-preparation stage averaged 7.425 ms (2.6% of 288.564 ms measured total),
below the required >=5% end-to-end gain even if fully removed. #26 documents
this decision without fabricating candidate evidence.

## Owned files

No implementation file is authorized yet. At the review checkpoint, list the
single smallest source/test file set required by the approved hypothesis in this
issue before assignment. Change only those files plus the three named candidate
result outputs and their metadata. Do not widen scope without a new review.

## Non-goals

No speculative tuning, bundled unrelated refactors, model/quality fix, new
framework, cloud service or dashboard. Do not alter fixtures, baseline artifacts,
or independent expected labels. Do not keep a change that fails the gates.

## Experiment and acceptance

- Change one variable. Preserve the approved model/profile/tokenizer/fixture set,
  provider, build settings, machine and API/CLI/JSON contracts.
- Run three fresh sessions per variant with the accepted release harness and
  correctness checks. Retain per-session raw samples, exact source/binary identity,
  asset hashes, and result paths. Run order is part of the evidence.
- Retain only with at least 5% lower pooled multi-window p50 and each of the three
  fresh-session median comparisons improving. Confirm in a new run with candidate/
  baseline order reversed.
- No other fixture may have a reproducible median regression above 5%. Investigate
  and repeat any p95, initialization, or peak-RSS regression above 10%. An
  unexplained reproducible regression beyond these limits blocks retention.
- Preserve exact accepted span/family/placeholder IDs, placeholder identity map,
  and masked text. Compare confidence separately; investigate all drift. Independently
  report the known label misses and wrong-family outputs.
- If evidence is ambiguous or negative, revert the candidate runtime change and
  retain its raw evidence with a clear no-optimization conclusion.
- Run targeted correctness checks, then `cargo clippy --all-targets --all-features
  -- -D warnings`, `cargo test --locked`, and manual model fixtures. Coordinate
  all builds/tests with root; never run a benchmark concurrently.

## Commands

Use the exact commands and comparison rules in `performance/README.md`. Example
outputs must use fresh paths:

```sh
uv run --no-project python scripts/benchmark.py --output performance/results/candidate.json --label candidate
uv run --no-project python scripts/benchmark.py --compare performance/results/validated-baseline.json performance/results/candidate.json
uv run --no-project python scripts/benchmark.py --output performance/results/confirmation.json --label confirmation
```

Build each variant from a reproducible source snapshot and ensure the executed
binary hash matches its result metadata. Preserve raw before/after results; a trace
run is separate and excluded from latency/RSS comparison. Semantic and trace
protocols: `docs/performance-correctness.md`, `docs/performance-profiling.md`.

## Handoff

Return the approved hypothesis, changed paths, variant order, commands, raw result
files, p50/p95/init/throughput/RSS and all fixture regressions, semantic and
confidence findings, decision against every gate, and rollback status. #26 waits
for this decision.
