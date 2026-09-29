Spec for [GitHub issue #26](https://github.com/agmxyz/tiktag/issues/26), child of
[#21](https://github.com/agmxyz/tiktag/issues/21).

## Goal and dependency

After #25 decides retain/reject, turn the validated raw evidence into concise
findings and a reproduction path. Natively blocked by #25. Existing draft results
must remain clearly labeled and must not be silently presented as validated.

## Reviewed no-change resolution

The user reviewed #24's validated baseline and the one tokenizer-reuse proposal,
then chose no runtime change. #25 was closed as not planned with no candidate
implementation or comparison run. No candidate or reversed-order confirmation
artifacts exist. State this explicitly; do not invent a measured speedup or claim
the retention gates passed. Report why the proposed stage's 2.6% share could not
meet the >=5% end-to-end gate even if fully removed.

## Owned files and non-goals

Own `docs/performance-report.md` and the final reproduction/summary portions of
`performance/README.md`. Link to raw artifacts; do not edit result data, source,
fixtures, or other issue specs. No new benchmark run without root's schedule; no
publication to external services.

## Report acceptance

- State setup, commands, hardware/OS, Rust/build/runtime/provider/thread/window
  settings, asset/fixture hashes, warm-ups, sample counts, p50/p95 definitions,
  throughput units and process-cold versus disk-cold distinction.
- Link the validated machine-readable baseline and separate native trace. State
  that candidate and confirmation artifacts do not exist after the reviewed
  no-change decision.
  Summarize end-to-end median/p95, initialization/first call, throughput, peak RSS,
  stages, native operators/provider placement, and every material regression.
- Explain which results are uninstrumented latency versus stage samples and native
  trace. State native profiling durations are aggregate operator work for recorded
  trace calls, not TikTag single-call wall time. State process RSS and CoreML memory
  boundaries.
- Separate behavior-preservation result from independent label quality. Include
  misses, partial/spurious/wrong-family masks without claiming production accuracy.
- Explain the agreed >=5%/all-three/reversed-order gate, <=5% other-fixture median
  regression bound, and >10% investigate/repeat triggers. They were not exercised
  because no candidate was run. State the user-approved no-optimization decision.
- Record tail noise, host limits, cache/power/load controls and unknowns. Give one
  justified next experiment only if evidence supports it.
- Keep README short enough for another developer to reproduce; link the detailed
  report and profiling/correctness protocols. Use `uv` for every Python command.

## Reproduction commands

Use new output names and the exact accepted harness:

```sh
uv run --no-project python scripts/benchmark.py --output performance/results/validated-baseline.json --label validated-baseline
uv run --no-project python scripts/benchmark.py --output performance/results/validated-trace.json --label validated-trace --trace --fixture multi_window --samples 2 --warmup 1 --repeats 1
# No candidate result exists; do not run --compare for this decision.
```

If a model bundle must be downloaded, verify its hashes before calling the result
comparable. Upstream downloads may change. See
`docs/performance-profiling.md` and the official
[ONNX Runtime profiling guide](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html);
quality/regression rules are in `docs/performance-correctness.md`.

## Handoff

Root reviews the report for defensible wording and confirms the linked artifacts
and reproduction commands. Do not close #21 or this issue from the report task.
