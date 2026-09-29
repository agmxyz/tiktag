# Local ONNX performance results

## Decision

No runtime change. After reviewing the baseline, the user chose not to pursue
the tokenizer-reuse experiment: window preparation averaged 7.425 ms, just
2.6% of the 288.564 ms instrumented pipeline total, below the 5% end-to-end
gain gate even if eliminated entirely. Issue #25 is closed as `NOT_PLANNED`.
No candidate or confirmation run exists, so optimization gates were not tested
and no speedup is claimed.

## Baseline

The [results archive](../performance/results/onnx-performance-results.zip)
contains all raw result files plus `performance/results/MANIFEST.sha256`.
The validated baseline member,
`performance/results/validated-baseline.json`, contains all samples,
environment and build metadata. It covers six synthetic fixtures, three fresh
processes per fixture, five warmups and 30 uninstrumented warm calls per
process. The table pools 90 calls per fixture; p95 is nearest rank. “Fresh”
means a new process, not a cold machine or disk.

| Fixture | Windows | Warm p50 / p95 (ms) |
| --- | ---: | ---: |
| short | 1 | 3.720 / 4.098 |
| near_limit | 1 | 67.250 / 70.735 |
| boundary | 2 | 110.647 / 126.545 |
| multi_window | 4 | 280.222 / 314.346 |
| unicode_repeated | 1 | 5.969 / 7.264 |
| no_entity | 1 | 14.588 / 15.490 |

Collected on Apple M2 Pro, macOS 26.6.2 arm64, Rust 1.88.0, locked `ort`
2.0.0-rc.12 and native ONNX Runtime 1.24.2 Release. The exact profile, source,
build settings and per-process results are recorded in the JSON. Linux is
unverified.

## Inputs and integrity

Comparable reruns require matching SHA-256 hashes for the profile, model and
fixtures. A download with different hashes is only a smoke test. The validated
baseline also records source and binary hashes.

| Input | SHA-256 |
| --- | --- |
| `models/profiles.toml` | `8a5ccbf017010aa67c1a3547d6e4f932b7cb6516ef23e06705c6732af2928654` |
| `models/distilbert-base-multilingual-cased-ner-hrl/config.json` | `38847be4dc6699b1218a749ed69f888c2ccc7b4deba98e3c4a1cac8cb34d54c8` |
| `models/distilbert-base-multilingual-cased-ner-hrl/tokenizer.json` | `bf1b59b7b11c95f194f51708d918eea378e09d05f84c0e1656dc5180e8117088` |
| `models/distilbert-base-multilingual-cased-ner-hrl/onnx/model_quantized.onnx` | `24a0b98f4dd4cd92842f5a541272f86f760225a64a29928eddef14bdb2edb986` |
| `performance/fixtures/boundary.json` | `3344f07fe242c28ad717e6527587119ca52b185838f81ad6b0cf9579b11156be` |
| `performance/fixtures/multi_window.json` | `f73d10ae82485b1d4bcb6af514c0ef66bbd923ec9b53aeb1e30b0bb849267b57` |
| `performance/fixtures/near_limit.json` | `e8601b525f61f44f37e5291ef49ff935db3f573c82cc16dda9c418714adbff88` |
| `performance/fixtures/no_entity.json` | `f93293407e5d934f37147030881fb7f7eba8d41a4e4fbed00ae61aa6722c8e31` |
| `performance/fixtures/short.json` | `038b591e3ea8c291730d6ce111e70bf44e4bf2b9d4dadd43514a2124aa4f8906` |
| `performance/fixtures/unicode_repeated.json` | `4ce916fc9caff0f3856fdbdfa747f42387a5e3395229e87e571ccf8fa380aa12` |

## Correctness and model quality

These are separate results:

- **Behavior regression check:** all 18 baseline rows match the fixture
  snapshots for spans, families, placeholders, digests and window counts.
  Sequence lengths and cross-session confidence values also match. No candidate
  was compared.
- **Independent synthetic labels:** `multi_window` has seven wrong-family masks
  (seven misses); `unicode_repeated` has two misses. No partial or spurious
  outputs; other fixtures match their labels, including `no_entity` at 0/0.
  These labels do not estimate production accuracy.

## Separate native ORT trace

The trace summary member is
`performance/results/validated-trace.json`; the [archive](../performance/results/onnx-performance-results.zip)
also contains the raw native trace member
`performance/results/validated-trace-0-multi_window_2026-09-29_16-51-51.json`.
They are from a separate instrumented run; its latency and RSS are excluded
from the uninstrumented baseline. Nine calls × four windows produced 36 window
executions. Node events sum to 12,564 CPU events (2,240,655 µs) and 972 CoreML
events (266,764 µs).
Largest CPU totals: MatMulInteger 981,438 µs, MatMul 378,210 µs, Where
251,831 µs.

These are accumulated operator durations, not wall time or exclusive shares;
overlap and parallel execution can make sums exceed elapsed time. Provider
labels describe only recorded Node events. They do not prove placement for all
executions or report every device kernel. Provider profiling depends on SDK
support, as described in the [ONNX Runtime profiling guide](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html).

## Raw artifacts

The ZIP contains 34 tracked result files, including earlier harness smoke
outputs and all validated baseline and trace files. Only `validated-*` files
support the results above. The manifest lists SHA-256 for each original file;
it does not hash itself. Archive SHA-256:
`94a96fa83aa258fbceba26ac4517bbd8df2be9cb4bc837de7f9d78a080307930`.

Extract and verify from the repository root:

```sh
mkdir -p /tmp/tiktag-results
unzip -q performance/results/onnx-performance-results.zip -d /tmp/tiktag-results
(cd /tmp/tiktag-results && shasum -a 256 -c performance/results/MANIFEST.sha256)
```

## Reproduction

From repository root, require Rust/Cargo, `uv`, `/usr/bin/time` and the exact
local model bundle. The driver builds the locked release harness before each
collection. Run one command at a time; never overlap collection with a build,
model test or other benchmark. Checks run after collection.

```sh
# Only if the model bundle is missing; verify every hash above afterward.
cargo run --locked --release -- download

# Uninstrumented warm latency; use a new output path for each run.
uv run --no-project python scripts/benchmark.py \
  --output performance/results/repro-baseline.json --label repro-baseline \
  --samples 30 --warmup 5 --repeats 3

# Separate native trace; its latency and RSS are diagnostic only.
uv run --no-project python scripts/benchmark.py \
  --output performance/results/repro-trace.json --label repro-trace --trace \
  --fixture multi_window --samples 2 --warmup 1 --repeats 1

# Run outside the benchmark window.
cargo test --locked
cargo clippy --all-targets --all-features -- -D warnings
cargo test --locked --release fixture_regression_synthetic_pipeline -- --ignored
```

The trace command writes timestamped raw ORT JSON beside its summary. Keep
uninstrumented timing and native tracing as separate runs.
