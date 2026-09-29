# Local ONNX benchmark

**Status:** validated baseline; no runtime optimization. #25 closed as NOT_PLANNED after review. No candidate or confirmation run exists.

- Detailed results, hashes, limits, and decision: [performance report](../docs/performance-report.md).
- Collection protocol: [profiling protocol](../docs/performance-profiling.md).
- Semantic and label rules: [correctness contract](../docs/performance-correctness.md).
- Plan and issue sequence: [performance plan](../docs/performance-plan.md), parent [#21](https://github.com/agmxyz/tiktag/issues/21), children [#22](https://github.com/agmxyz/tiktag/issues/22)–[#26](https://github.com/agmxyz/tiktag/issues/26).

## Reproduce

Requirements: Rust/Cargo, uv, /usr/bin/time, and the exact local model bundle. Linux has not been measured. From repository root, preserve the bundle and verify its hashes against the [report](../docs/performance-report.md). A changed upstream download is a smoke test, not comparable evidence.

Run each collection alone. Do not overlap it with another benchmark, build, or model test.

```sh
# Only if the bundle is missing; verify all hashes after download.
cargo run --locked --release -- download

# Default six-fixture protocol, specified explicitly.
uv run --no-project python scripts/benchmark.py \
  --output performance/results/repro-baseline.json --label repro-baseline \
  --samples 30 --warmup 5 --repeats 3

# Separate native trace; do not use its latency or RSS for comparison.
uv run --no-project python scripts/benchmark.py \
  --output performance/results/repro-trace.json --label repro-trace --trace \
  --fixture multi_window --samples 2 --warmup 1 --repeats 1

# Correctness checks; run outside the benchmark window.
cargo test --locked
cargo clippy --all-targets --all-features -- -D warnings
cargo test --locked --release fixture_regression_synthetic_pipeline -- --ignored
```

The baseline and native trace are separate. The benchmark driver builds the locked release harness; use a fresh output path each time. The trace command writes its raw ORT JSON beside the summary. Original results/baseline.json and results/ort-profile.json remain unaccepted drafts; use the validated artifacts linked from the report.
