Parent spec for [GitHub issue #21](https://github.com/agmxyz/tiktag/issues/21).
This local file mirrors the self-contained GitHub issue body.

## Status

#22 and #23 passed delegated audits and checks; #24 collected the validated
baseline and separate trace. After reviewing one concrete experiment, the user
chose no runtime change; #25 was closed as not planned without a candidate run.
#26 completed the final report. Earlier `baseline.json` and `ort-profile.json`
remain draft evidence; validated results use new paths.

## Objective and scope

Establish reproducible local ONNX performance evidence for TikTag, then retain one
small measured optimization only if it meets the gates below. Primary workload:
reused-library four-window calls. Initialization, first-call and short-input
latency are guardrails.

Measure initialization, first call, warm latency and stages; record median/p95,
sample count, throughput units, window/token counts, peak process RSS, runtime and
asset provenance. Use committed synthetic fixtures, release builds, warm-ups, and
fresh sessions. Run native ONNX Runtime profiling separately from latency runs.
Report independent label misses and incorrect masks separately from behavior
preservation. No input text or detected entity values in logs/results.

No cloud, dashboard, Langfuse, inference framework migration, or speculative
threading, batching, quantization, or GPU work. A no-optimization result is valid.

## Ownership and non-goals

Root owns coordination, issue maintenance and exclusive benchmark scheduling. This
parent issue owns no implementation files; implementation belongs only to the
bounded child scopes. Do not perform unreviewed source changes or delegate #25
before the baseline-review gate.

## Agreed gates

- Same pinned assets/provider: preserve accepted spans/families, placeholder IDs
  and identity map, and masked text. Report confidence drift separately and
  investigate it. Keep known model-quality misses visible; fixing them is separate.
- At least 5% lower pooled multi-window median, with each of the three fresh-session
  comparisons improving; confirm in a rerun reversing baseline/candidate order.
- No reproducible median regression over 5% on another fixture.
- A p95, initialization or peak-RSS regression over 10% triggers investigation and
  repetition. An unexplained, reproducible regression beyond the limits blocks
  retention. These are practical gates, not statistical-significance claims.
- User reviews the validated baseline and exactly one experiment proposal before
  issue #25 is assigned or optimization implementation begins.
- Performance claims cover the recorded Apple M2 Pro Mac only. Linux remains
  unverified until measured.

## Child issues and native dependency graph

| Issue | Work | Local spec | Dependencies |
| --- | --- | --- | --- |
| [#22](https://github.com/agmxyz/tiktag/issues/22) | Instrumentation and harness audit | `docs/performance-issues/01-harness.md` | None |
| [#23](https://github.com/agmxyz/tiktag/issues/23) | Fixture labels and semantic regression guards | `docs/performance-issues/02-correctness.md` | None |
| [#24](https://github.com/agmxyz/tiktag/issues/24) | Validated baseline and separate native trace | `docs/performance-issues/03-baseline.md` | Blocked by #22 and #23 |
| [#25](https://github.com/agmxyz/tiktag/issues/25) | One optimization experiment | `docs/performance-issues/04-experiment.md` | Blocked by #24 and user review |
| [#26](https://github.com/agmxyz/tiktag/issues/26) | Final report and reproduction guide | `docs/performance-issues/05-report.md` | Blocked by #25 |

#22 and #23 may run in parallel with disjoint file ownership. Root controls
benchmarks and model-dependent checks so they do not overlap builds, tests, or
other benchmark runs. Child issues and dependencies already exist in GitHub; update
them in place, do not create duplicates. User review is a human gate that GitHub
cannot represent as a dependency.

## Commands and references

The executable reproduction commands are maintained in `performance/README.md`.
Run Python only through `uv`, e.g.
`uv run --no-project python scripts/benchmark.py --output performance/results/run.json`.
Native profiling protocol: `docs/performance-profiling.md` and the official
[ONNX Runtime profiling guide](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html).
Correctness protocol: `docs/performance-correctness.md`. Issue operations follow
`docs/agents/issue-tracker.md`.

## Acceptance

Close this parent only when #22–#26 have been completed or explicitly resolved,
the validated baseline and trace raw artifacts are linked from the final report,
the report explicitly states that no candidate or confirmation artifacts exist
after the reviewed no-change decision, and the final report states limitations
and reproduction steps. Name a next experiment only if evidence justifies one.
