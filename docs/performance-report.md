# Local ONNX performance report

## Decision

The user reviewed the validated #24 baseline and chose no runtime change. #25 is closed as NOT_PLANNED. No candidate implementation, candidate benchmark, or reversed-order confirmation exists. The agreed optimization gates were not exercised; no speedup is claimed.

The tokenizer-reuse idea had a 7.425 ms mean window-preparation stage against 288.564 ms mean instrumented pipeline time (2.6%). Even eliminating that entire stage could not reach the 5% end-to-end gate. No next experiment is justified by this evidence within the agreed scope.

## Evidence and provenance

- [Validated baseline](../performance/results/validated-baseline.json): complete, 18 rows; six fixtures × three fresh sessions. Each session has five warmups and 30 ordinary warm samples per fixture.
- [Trace summary](../performance/results/validated-trace.json) and [raw native ORT trace](../performance/results/validated-trace-0-multi_window_2026-09-29_16-51-51.json): separate instrumented run.
- Earlier results/baseline.json and results/ort-profile.json remain unaccepted drafts. Use the validated artifacts above.
- SHA-256: baseline e82b9ab2024ba3f8d68b1704346b24369f286f86f2a75a9b03bdca98b5573033; trace summary e90678dc620bf70af912ebc6a947b1e13232a0a733a1c0544928c4a938aaf3b8; native trace f8255b703176d59ba1041ac58ad0316b17cf0db665626f24180dd0ac499f964d.
- Host: Apple M2 Pro, 16 GiB reported memory, 10 logical CPUs; macOS 26.6.2 arm64. Linux is unverified.
- Toolchain/runtime: Cargo and rustc 1.88.0; locked Rust binding ort 2.0.0-rc.12; native ORT 1.24.2 Release, commit 058787c.
- Build: cargo build --locked --release --features profiling --example pipeline_bench. Binary SHA-256: a8c1c82ed4a4d981dea7bfcf488f7542f5c1a723e6cb289e786bade5023ec155. Cargo.lock SHA-256: 2d65083e72227af0eb3919e965def8d2fec4881374885d050f505eb92253a61b.
- Run source: commit 1f09449b2227a030bf4a6fbb2861014d64081c26; source snapshot SHA-256 aaa452bd0a7068c4339c55ff73ae3876327396135e79bf2024f918dd2bd8f056. At collection time, source/results were uncommitted; the recorded source snapshot hashes identify the measured files. All 20 recorded source hashes match current files.
- Profile: models/profiles.toml, model bundle under models/distilbert-base-multilingual-cased-ner-hrl/, max_tokens=512, overlap_tokens=128, email recognizer enabled. CoreML registered with CPU fallback; registration alone does not establish placement. Execution was sequential, graph optimization Level3, ORT inter-op default, intra-op default (0). Native profiling was off for baseline warm samples; the profiling Cargo feature enabled stage sampling.
- Captured build/runtime override variables were unset; RUST_LOG=warn. See the baseline JSON for the exact environment fields.
- Rehashed profile, model files, and fixtures match the baseline metadata.

| Pinned input | SHA-256 |
| --- | --- |
| models/profiles.toml | 8a5ccbf017010aa67c1a3547d6e4f932b7cb6516ef23e06705c6732af2928654 |
| models/distilbert-base-multilingual-cased-ner-hrl/config.json | 38847be4dc6699b1218a749ed69f888c2ccc7b4deba98e3c4a1cac8cb34d54c8 |
| models/distilbert-base-multilingual-cased-ner-hrl/tokenizer.json | bf1b59b7b11c95f194f51708d918eea378e09d05f84c0e1656dc5180e8117088 |
| models/distilbert-base-multilingual-cased-ner-hrl/onnx/model_quantized.onnx | 24a0b98f4dd4cd92842f5a541272f86f760225a64a29928eddef14bdb2edb986 |
| fixture boundary.json | 3344f07fe242c28ad717e6527587119ca52b185838f81ad6b0cf9579b11156be |
| fixture multi_window.json | f73d10ae82485b1d4bcb6af514c0ef66bbd923ec9b53aeb1e30b0bb849267b57 |
| fixture near_limit.json | e8601b525f61f44f37e5291ef49ff935db3f573c82cc16dda9c418714adbff88 |
| fixture no_entity.json | f93293407e5d934f37147030881fb7f7eba8d41a4e4fbed00ae61aa6722c8e31 |
| fixture short.json | 038b591e3ea8c291730d6ce111e70bf44e4bf2b9d4dadd43514a2124aa4f8906 |
| fixture unicode_repeated.json | 4ce916fc9caff0f3856fdbdfa747f42387a5e3395229e87e571ccf8fa380aa12 |

## Uninstrumented baseline

Warm latency uses ordinary anonymize calls. Each fixture pools three sessions × 30 calls. p50 is the median; p95 is nearest-rank. Initialization, first-call, and throughput values are medians across the three sessions. Throughput is sequential and shown as documents/s, input bytes/s, and original input tokens/s. RSS is the range of the three whole-child high-water readings in decimal MB.

| Fixture (windows) | Warm p50 / p95 ms (n=90) | Init p50 ms | First-call p50 ms | Throughput median: docs/s · bytes/s · tokens/s | Peak RSS range MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| short (1) | 3.720 / 4.098 | 340.411 | 10.408 | 272.770 · 11,184 · 2,727.7 | 411.5–426.2 |
| near_limit (1) | 67.250 / 70.735 | 376.830 | 79.583 | 14.765 · 29,693 · 7,426.9 | 492.8–495.9 |
| boundary (2) | 110.647 / 126.545 | 344.551 | 126.500 | 8.741 · 21,537 · 5,384.2 | 537.4–570.5 |
| multi_window (4) | 280.222 / 314.346 | 333.992 | 289.948 | 3.544 · 22,741 · 5,621.5 | 568.0–578.9 |
| unicode_repeated (1) | 5.969 / 7.264 | 330.417 | 12.598 | 169.485 · 16,948 · 4,745.6 | 414.6–415.0 |
| no_entity (1) | 14.588 / 15.490 | 338.015 | 21.804 | 68.364 · 26,662 · 7,314.9 | 427.4–428.6 |

Four-window per-session p50 values were 309.992, 272.977, and 279.452 ms; per-session p95 values were 336.914, 281.140, and 311.837 ms. The pooled p50 was 280.222 ms. With three sessions and adjacent calls that may be correlated, these tail and session spreads are descriptive, not a stable tail estimate.

Initialization measures Tiktag construction after process start. First-call timing includes lazy work. “Fresh” means a new process/session, not a cold disk or machine; file caches were not flushed. Build time and process startup are excluded.

## Stage timings and native trace

Stage samples are five additional measured calls per baseline session, excluded from the ordinary latency table. For the four-window fixture, arithmetic means across 15 stage samples were:

| Diagnostic stage | Mean |
| --- | ---: |
| window preparation | 7.425 ms |
| model execution | 277.165 ms |
| inner pipeline total | 288.564 ms |

The total is the instrumented inner pipeline timer. Stage subdivisions are diagnostic and may not sum to total. They are separate from ordinary end-to-end latency.

The separate native trace ran nine calls: first call, one warmup, two requested samples, and five stage calls. Four windows per call yielded 36 window executions. Its 13,536 Node kernel-time events reported 12,564 CPU events (2,240,655 µs summed duration) and 972 CoreML events (266,764 µs). Largest CPU operator totals were MatMulInteger (1,332 events, 981,438 µs), MatMul (432, 378,210 µs), and Where (216, 251,831 µs).

These are accumulated operator durations across the trace, not single-call wall time or exclusive shares; parallel or overlapping work can make sums exceed elapsed time. Provider labels describe this trace’s Node events only. They do not prove placement for every execution or report every device kernel. The [ONNX Runtime profiling guide](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html) notes that execution-provider profiling depends on support in that provider’s SDK. Trace-run latency and RSS are excluded from the baseline.

## Correctness and limits

All 18 baseline rows match the #23 snapshot for replacement spans, families and placeholders; replacement and map digests; anonymized-text digest; and window count. Sequence lengths match recorded fixture token counts. Within-run confidence drift is zero, and confidence lists match across the three sessions.

Independent labels are separate from behavior checks. multi_window has 17/24 exact labels and seven wrong-family masks (also seven misses); unicode_repeated has 3/5 exact labels and two misses. There are no partial or spurious outputs. Other fixtures match all expected labels; no_entity is 0/0. These synthetic labels do not estimate production accuracy. No candidate was compared with this behavior snapshot.

Peak RSS uses /usr/bin/time child high-water memory. It includes post-timing fixture validation and a second tokenizer loaded by the harness; it is not library-only attribution. A CoreML service may use out-of-process memory excluded from this measurement. The host was on AC at 100% charge, low-power mode off, with no competing build, model test, or benchmark at collection start. Thermal state and external-load telemetry were not recorded. Results apply to this macOS host.

## Gates and reproduction

A candidate would need at least 5% pooled multi_window median improvement, improvement in all three fresh-session comparisons, and a confirming run in reversed baseline/candidate order. No other fixture may have a reproducible median regression above 5%. A p95, initialization, or peak-RSS regression above 10% triggers investigation and a repeat; unexplained reproducible regressions block retention. Behavior must preserve the pinned semantic output. These practical gates were not exercised because no candidate exists.

From the repository root, keep the host idle and run each collection separately. Do not overlap benchmarks with builds, model tests, or other benchmarks. Verify the local asset hashes above before treating a rerun as comparable. If assets are missing, a download is comparable only if all hashes still match.

```sh
# Only if the local bundle is missing; verify hashes after download.
cargo run --locked --release -- download

# Six-fixture baseline: 3 sessions, 5 warmups, 30 ordinary warm samples.
uv run --no-project python scripts/benchmark.py \
  --output performance/results/repro-baseline.json --label repro-baseline \
  --samples 30 --warmup 5 --repeats 3

# Separate native trace; its timing and RSS are diagnostic only.
uv run --no-project python scripts/benchmark.py \
  --output performance/results/repro-trace.json --label repro-trace --trace \
  --fixture multi_window --samples 2 --warmup 1 --repeats 1

# Correctness and repository checks; run outside the benchmark window.
cargo test --locked
cargo clippy --all-targets --all-features -- -D warnings
cargo test --locked --release fixture_regression_synthetic_pipeline -- --ignored
```

The trace command writes a timestamped native ORT JSON beside its summary. No candidate comparison exists for this no-change decision. See the [profiling protocol](performance-profiling.md), [correctness contract](performance-correctness.md), and [issue plan](performance-plan.md).
