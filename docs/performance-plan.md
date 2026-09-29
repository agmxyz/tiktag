# TikTag local ONNX performance plan

Status: #22 and #23 passed their audits and checks; #24 produced a validated local
baseline and separate native trace. The user reviewed one experiment and chose
no runtime change; #25 is closed as not planned. #26 is unblocked and documenting
the result.
Earlier `baseline.json` and `ort-profile.json`
remain draft evidence and are not the validated results.

## Workflow and scope

The work follows the user-approved sequence: docs-backed grill, agreed design and
gates, bounded GitHub issues with dependencies, delegated work, then review of the
validated baseline before assigning an optimization.

Primary workload is reused-library multi-window inference. Constructor/first-call
startup and short text are regression checks. Collect release-build baseline and
one separately instrumented ONNX Runtime trace; retain a runtime change only when
repeatable end-to-end evidence supports it. A no-change conclusion is valid.

No cloud service, dashboard, Langfuse, inference-framework migration, or speculative
threading, batching, quantization, or GPU work is in scope. Do not log input text,
entity values, or reversible outputs.

## Settled decisions

- Correctness: same pinned model/profile/provider must preserve accepted spans and
  families, placeholder identities, and rewritten text. Report confidence drift
  separately and investigate it. Keep independent model misses and wrong-family
  masks visible; fixing model quality is separate work.
- Platform: performance claims apply to the recorded Apple M2 Pro Mac. Keep the
  harness portable, but mark Linux unverified until measured there. Provider
  registration is not proof of node placement.
- Retain a candidate only if pooled median latency improves by at least 5% on the
  multi-window workload, every one of three fresh-session comparisons improves,
  and a confirmation run in reversed baseline/candidate order agrees.
- No reproducible median regression above 5% on another fixture. A p95,
  initialization, or peak-RSS regression above 10% triggers investigation and a
  repeat; unexplained reproducible regression blocks retention. These are practical
  gates, not statistical-significance claims. Do not retain an ambiguous result.
- User review of the validated baseline and one proposed experiment is required
  before issue #25 is assigned or optimization implementation begins.

## Verified repository facts

- `tiktag` is a Rust library and thin CLI. A `Tiktag` instance loads the tokenizer
  and ONNX session once; subsequent calls reuse them.
- `Cargo.lock` pins `ort 2.0.0-rc.12`; the model profile sets `max_tokens = 512` and
  `overlap_tokens = 128`.
- The macOS build registers CoreML with CPU fallback. Native trace data is required
  to describe actual placement. CoreML service memory may be outside process RSS.
- Existing Criterion coverage is end-to-end warm-call timing. Existing decoder
  unit tests cover BIO spans/offsets; model fixture tests are ignored by default
  because they require local assets.
- Issue operations use `gh` and native dependencies per
  `docs/agents/issue-tracker.md`.

## Draft evidence, not accepted baseline

`performance/results/baseline.json` contains 18 completed runs: six fixtures, three
fresh processes per fixture, and 30 warm samples per process. For the four-window
fixture (1,586 input tokens; 1,976 tokens processed including overlap and special
tokens), pooled p50 is 277.123 ms and nearest-rank p95 is 334.518 ms (n=90). The
three per-session medians are 276.461, 281.534, and 276.356 ms; per-session p95s
are 304.410, 427.800, and 283.377 ms, with a 560.783 ms maximum sample. Treat the
tail as noisy.

Instrumented stage samples attribute a median-session 95.8% of total to
`Session::run`; window preparation is about 7.55 ms (2.71%). Multi-window
initialization totals are 330.786, 346.999, and 344.214 ms. Whole-child peak RSS is
546.6, 546.0, and 553.2 MB; it includes later harness validation and excludes
out-of-process CoreML memory.

`performance/results/ort-profile.json` is a separate trace run. It shows mixed
CPU/CoreML placement; CPU `MatMulInteger`, `MatMul`, and `Where` are the largest
aggregated operators. It covers nine inference passes, not one call. The recorded
ORT build is 1.24.2. See `docs/performance-profiling.md` for API and trace limits.

These values are investigation context only. They were collected before the
workflow correction and have since been audited and rerun after issues #22 and #23. Existing
independent labels show seven wrong-family spans among 24 multi-window targets and
two misses among five Unicode targets; this is not production accuracy evidence.

## Validated local baseline and checkpoint

The accepted raw run is `performance/results/validated-baseline.json`: six fixtures,
three fresh sessions each, five warm-ups and 30 ordinary warm samples per session.
For `multi_window`, pooled p50 is 280.222 ms and nearest-rank p95 is 314.346 ms
(n=90); session medians are 309.992, 272.977, and 279.452 ms. Mean instrumented
model execution is 277.165 ms of 288.564 ms total (96.1%). The separate trace is
`performance/results/validated-trace.json` with its raw native JSON. It covers nine
calls and 36 window executions and reports mixed CPU/CoreML Node events. Traced
timings and RSS are excluded from latency comparisons. #23's semantic snapshot
matches all 18 baseline rows; confidence drift is zero. Independent label quality
still has seven wrong-family masks on `multi_window` and two Unicode misses.

The reviewed single experiment was to reuse a preconfigured window tokenizer
instead of cloning/configuring it per long call. The entire window-preparation
stage averages 7.425 ms (2.6% of measured total), below the >=5% end-to-end
retention threshold even if fully removed. The user chose no runtime change;
no candidate or confirmation runs exist, and #25 was explicitly resolved.

## Issue index and dependencies

Local specifications below are repository files. GitHub issue bodies remain
self-contained and name these repository-relative paths as spec references.

| Issue | Scope | Local spec | Dependency |
| --- | --- | --- | --- |
| [#21](https://github.com/agmxyz/tiktag/issues/21) | Parent plan and review gates | `docs/performance-issues/parent.md` | — |
| [#22](https://github.com/agmxyz/tiktag/issues/22) | Instrumentation and harness audit | `docs/performance-issues/01-harness.md` | None; parallel with #23 |
| [#23](https://github.com/agmxyz/tiktag/issues/23) | Fixture labels and masking regression checks | `docs/performance-issues/02-correctness.md` | None; parallel with #22 |
| [#24](https://github.com/agmxyz/tiktag/issues/24) | Validated baseline and native profile | `docs/performance-issues/03-baseline.md` | Blocked by #22 and #23 |
| [#25](https://github.com/agmxyz/tiktag/issues/25) | One measured optimization | `docs/performance-issues/04-experiment.md` | Blocked by #24 and user review |
| [#26](https://github.com/agmxyz/tiktag/issues/26) | Final report and reproduction guide | `docs/performance-issues/05-report.md` | Blocked by #25 |

Native GitHub child and dependency links were verified. The body of #25 must keep
the user-review checkpoint explicit even though GitHub cannot encode that review as
an issue dependency. Root coordinates issue flow and benchmark scheduling; no
benchmarks may overlap another benchmark, model test, or build-heavy check.

## Documentation and execution references

- Reproduction and draft-data caveats: `performance/README.md`.
- Native profiling details: `docs/performance-profiling.md`, grounded in the
  [official ONNX Runtime profiling guide](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html).
- Fixture quality and semantic regression rules: `docs/performance-correctness.md`.
- All Python commands must run through `uv`, for example
  `uv run --no-project python scripts/benchmark.py ...`.
- #24's exclusive measurement window and user review are complete. #25 is closed
  as not planned; #26 is the final documentation step.
