Spec for [GitHub issue #24](https://github.com/agmxyz/tiktag/issues/24), child of
[#21](https://github.com/agmxyz/tiktag/issues/21).

## Goal

Collect the first accepted, reproducible baseline after #22 and #23 pass. Existing
raw files are unapproved draft evidence. Preserve them; write to new result paths.
Root owns the exclusive measurement schedule.

## Owned outputs

- `performance/results/validated-baseline.json` and child stderr files
- `performance/results/validated-trace.json`, native trace, and trace stderr file
- `docs/performance-plan.md` is root-owned; do not edit it. Do not change source,
  fixtures, harness, or issue bodies in this task.

## Prerequisites and non-goals

Blocked by #22 (instrumentation/harness audit) and #23 (label and semantic checks).
Use pinned assets and the validated harness. No code change, optimization,
concurrent build/model test/benchmark, disk-cache flush claim, or performance claim
for another host.

## Collection

- Default run: all six fixtures, three fresh sessions per fixture, five warm-ups
  and 30 uninstrumented warm calls/session. Preserve all 18 raw rows/samples.
- Record constructor, first call, p50/p95 and n, input/processed tokens, windows,
  throughput with units, stage means, peak/current RSS and boundary conditions.
- Capture native ORT profiling separately from timing. Report registered provider
  and actual per-node placement distinctly; aggregate `Node/*_kernel_time` by
  operator/provider with event count and duration sums. State trace call/window
  count; do not interpret aggregate operator time as one-call wall latency.
- Record Rust and native ORT versions, build flags, OS/hardware, profile/model/
  tokenizer/fixture hashes, execution provider, thread/window settings and relevant
  environment. Note power/load conditions and unknowns.
- Report independent expected-label misses and incorrect masks from #23.
- Explain the measured bottleneck, one small falsifiable experiment proposal,
  tradeoffs, potential gain and rollback. Present it to the user and stop. Do not
  assign or begin #25 until the user reviews the validated baseline and proposal.

## Commands

Run from repository root only in the time slot root assigns. Python is always run
through `uv`:

```sh
# Only if the expected bundle is missing. Then check hashes before measuring.
cargo run --locked --release -- download
uv run --no-project python scripts/benchmark.py --output performance/results/validated-baseline.json --label validated-baseline
uv run --no-project python scripts/benchmark.py --output performance/results/validated-trace.json --label validated-trace --trace --fixture multi_window --samples 2 --warmup 1 --repeats 1
```

The default run must resolve to six fixtures × three sessions × 30 warm samples.
The trace is separate and its latency/RSS must not enter comparison. Compare no
unlike runs. Verify downloaded model/config/tokenizer/profile hashes against the
recorded bundle; upstream downloads can change and then are not comparable.
Follow `performance/README.md`, `docs/performance-profiling.md`, and the official
[ONNX Runtime profiling guide](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html).

## Acceptance and handoff

Complete metadata and sample counts; preserve raw artifacts; run the correctness
gate; document noisy tails, RSS boundary and provider visibility; summarize stages
and operators without overstating them; present one proposal for user review.
Return artifact paths, exact commands, hashes/runtime/machine, stage and operator
findings, fixture errors, limitations, and the proposed experiment. #25 remains
blocked by this issue and the human review checkpoint.
