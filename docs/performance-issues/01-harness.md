Spec for [GitHub issue #22](https://github.com/agmxyz/tiktag/issues/22), child of
[#21](https://github.com/agmxyz/tiktag/issues/21).

## Goal

Audit the optional pipeline instrumentation and make the benchmark evidence
comparable. Preserve inference, windowing, masking, CLI, and JSON behavior.

## Owned files

- `Cargo.toml`, `src/measurement.rs`, `src/lib.rs`, `src/runtime.rs`
- `examples/pipeline_bench.rs`, `scripts/benchmark.py`
- `docs/performance-profiling.md` (keep API facts and measurement boundaries
  aligned with implementation)

Coordinate fixture schema with #23. Do not edit its fixture or test files, root
planning/issue docs, or optimization code. Do not collect the full baseline here.

## Non-goals

No model, decoder, overlap-resolution or anonymization behavior change; no
performance optimization; no new inference framework or profiling dependency
without a documented reason; no PII or reversible-output logging.

## Acceptance

- Verify actual timer boundaries and make instrumentation opt-in. Ordinary
  `Tiktag::new`/`anonymize` must not collect stage samples or log input/entities.
  Keep ordinary warm samples uninstrumented; run stage timing and native tracing
  separately. Check timing-gate overhead and output equality.
- Keep constructor and tokenizer/session sub-times, first-call latency, input
  tokens/windows, per-stage timings, throughput units, raw warm samples, nearest
  rank p95 and sample count. Preserve stderr on child failures and reject partial,
  duplicate, invalid or incomplete results.
- Comparison output must show pooled and per-session p50/p95 per fixture, plus the
  three matched baseline/candidate session-median changes used by #25. Report all
  regressions; do not let a pooled gain conceal a session or fixture regression.
- Comparison must reject differences in fixture/model/tokenizer/profile hashes,
  OS/hardware, ORT build, provider configuration, thread/window/build settings,
  relevant environment, and sample/warmup/repeat contract. Candidate source/binary
  hashes are provenance and are expected to change.
- Compare semantic output separately from confidence: exact span/family,
  placeholder IDs and mapping, and masked text must match. Report confidence drift
  separately. Do not persist original entity values or reversible output.
- Respect Cargo target/build output overrides or reject them; never silently run a
  stale binary. Document that whole-child RSS includes validation and a second
  tokenizer after timing; CoreML service memory may be excluded.
- Keep existing draft result files unchanged. Harness changes require new validated
  results under new paths.

## Verification commands

From repository root, use:

```sh
cargo build --locked --release --features profiling --example pipeline_bench
cargo test --locked
cargo clippy --all-targets --all-features -- -D warnings
uv run --no-project python scripts/benchmark.py --output performance/results/harness-smoke.json --fixture short --samples 2 --warmup 1 --repeats 1
uv run --no-project python scripts/benchmark.py --output performance/results/harness-trace.json --trace --fixture multi_window --samples 2 --warmup 1 --repeats 1
```

Coordinate build/model access with root. Do not run the full six-fixture baseline;
#24 owns that exclusive measurement window.

The locked binding is `ort 2.0.0-rc.12`. Native profiling facts, separate-trace
protocol, and official source are in `docs/performance-profiling.md` and the
[ONNX Runtime profiling guide](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html).
Reproduction overview: `performance/README.md`.

## Dependency and handoff

No blockers; may run in parallel with #23. #24 is natively blocked by #22 and #23.
Return changed files, verification results, schema/comparison keys, profiling
limitations, and remaining blockers. Do not assign or begin #25; baseline review
is a separate user checkpoint.
