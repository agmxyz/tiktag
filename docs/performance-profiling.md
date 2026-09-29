# Local ONNX profiling protocol

Issues #22 and #23 audits are complete. Issue #24's validated baseline and
separate native trace are recorded in `performance/results/validated-baseline.json`
and `performance/results/validated-trace.json`. The older `baseline.json` and
`ort-profile.json` remain preliminary artifacts from before those audits.

## Verified runtime and pipeline

`Cargo.lock` pins the Rust binding `ort 2.0.0-rc.12`; this is distinct from the
ONNX Runtime native build reported by `ort::info()` (the validated baseline
reports ORT 1.24.2). The locked binding source is under
`$CARGO_HOME/registry/src/index.crates.io-*/ort-2.0.0-rc.12/`; this machine's
resolved checkout is
`/Users/mmo/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/ort-2.0.0-rc.12/`:

- `src/session/builder/impl_options.rs`: `SessionBuilder::with_profiling(path)`
  enables profiling and supplies the output path.
- `src/session/mod.rs`: `Session::end_profiling(&mut self)` flushes the trace and
  returns its filename. The binding says to call it explicitly; otherwise the
  profile file is empty.

Thus native profiling is supported without a new dependency. The Rust API and
the session setup in `src/runtime.rs` are compatible with the official
[ONNX Runtime profiling guide](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html),
which describes JSON operator/thread timing and notes that execution-provider
profiling depends on support in that provider's SDK.

Current pipeline boundaries in `src/runtime.rs` and `src/lib.rs`:

| Stage | Work included |
| --- | --- |
| Initialization | Profile and bundle validation, tokenizer load, label/config load, ORT initialization, provider/session creation. Constructor total includes all of these; tokenizer and session are sub-times. |
| Tokenization | Full-input encode used to obtain the unwindowed token count. |
| Window preparation | For long input: tokenizer clone/configuration, second encode with truncation/stride, overflow window collection. |
| Tensor preparation | Input ID, attention-mask, and optional token-type arrays. |
| Model execution | `Session::run` per encoding/window, including Rust input wrappers and return from ORT. |
| Decoding | Output extraction/shape checks, argmax/probabilities, and entity span decoding. |
| Stitching | Per-window emit-region calculation and cross-window span stitching/deduplication. |
| Recognizers | Enabled email recognizer. |
| Masking | Candidate filtering, overlap resolution, placeholder assignment, and text rewrite. |
| Total | Timed inner anonymization path, ending before result-struct construction. Stage times are diagnostic subdivisions; residual allocations, control flow, and timing overhead mean their sum need not equal total. |

The optional Cargo feature `profiling` exposes numeric timing through
`Tiktag::new_measured(profiles_path, trace_prefix)`,
`Tiktag::anonymize_measured(text)`, and `Tiktag::end_profiling()`. The ordinary
`Tiktag::new` and `Tiktag::anonymize` paths pass a disabled measurement gate;
stage clocks do not start and stage counters do not change. The harness checks
semantic output equality between ordinary and measured calls, while recording
confidence drift separately. Static inspection confirms the disabled path gets
`None` from `Measurement::start()` without calling `Instant::now()`, and
`add_elapsed(None)` leaves timing counters unchanged. The short smoke recorded
ordinary warm p50 3.522 ms and measured-stage `total_ms` median 3.477 ms
(-1.29%). This run-level difference is diagnostic only; it does not isolate the
disabled branch cost. Token-level debug logging was removed so ordinary use
cannot log input token values or decoded entities.

## Collection protocol

1. Use the same committed synthetic fixture, model/profile files, release
   settings, host, and environment for every comparison. Record fixture and
   asset hashes, Rust/native ORT versions, OS/hardware, build flags, registered
   provider configuration, thread settings, window configuration, and all
   relevant runtime/build environment overrides.
   The driver hashes the source files, including untracked Rust, example, and
   harness files, and records the exact executable Cargo reports after building.
   Cargo target-directory and target-triple overrides are honored; the driver
   never guesses the executable path. Source and binary hashes are provenance
   and may differ between variants.
2. Collect latency in a fresh process/session per fixture. Record constructor
   initialization and first-call latency separately. Warm the session, then
   measure repeated calls through ordinary `Tiktag::anonymize`; use these
   uninstrumented samples for median, nearest-rank p95, and throughput.
3. Collect stage timings in additional warm calls after the primary loop. Keep
   these samples out of the latency distribution. They explain stage cost; they
   are not the performance result.
4. Collect a native ORT trace in a separate process/run with the same fixture
   and assets. Enable profiling before session creation, make a recorded number
   of calls, then call `end_profiling`. Do not use trace-run latency or memory in
   baseline comparisons: tracing adds overhead and output data. Ignore all trace-run
   latency, stage timings, and memory values.
5. Retain raw samples and trace, plus the trace call count. Aggregate ORT `Node`
   events named `*_kernel_time` by `args.provider` and `args.op_name`, with event
   count and summed `dur` (microseconds). Keep unrecognized/missing provider values visible as
   `unknown`; do not infer placement from registration alone. Event-duration
   sums describe accumulated operator work over the trace, not TikTag wall time
   or exclusive shares; threading/overlap can make sums exceed wall duration.
   Provider values describe placement reported by this trace's Node events;
   they do not prove placement for every execution or account for all device
   kernel work. The official guide notes that execution-provider profiling
   detail depends on support in that provider's SDK.

The result schema records each warm sample, constructor and first-call timing,
stage samples, token/window counts, and throughput in documents, input bytes, and
input tokens per second. Comparisons require complete, unique fixture/session
rows and matching asset hashes, machine, ORT build, provider and thread settings,
environment, build settings, and sample contract. They report pooled and
per-session p50/p95 plus each matched session-median change. Source and binary
hashes are shown as provenance and are allowed to change.

Behavior snapshots contain span/family/placeholder tuples, hashes of the
replaced values and placeholder map, a hash of the exact masked text, sequence
length, and window count. They do not contain input, replaced values, or masked
text. Confidence scores are numeric and compared separately. Label-quality
counts and mismatches stay in a separate report from behavior regressions.
Incomplete runs remain in a `.partial` file; comparisons reject them.

The trace example uses `multi_window`, `--samples 2 --warmup 1`, plus five stage
samples. Including the first call, that is nine TikTag calls. At four windows
per call, the trace aggregates 36 `Session::run` window executions; its operator
totals are not single-call costs. For example, collect a trace separately with:

```sh
uv run --no-project python scripts/benchmark.py \
  --output performance/results/trace-review.json \
  --label trace-review --trace --fixture multi_window \
  --samples 2 --warmup 1 --repeats 1
```

The script uses a fresh output path. Treat any latency fields in this trace
result as instrumentation-contaminated; do not pass it to `--compare`.

## Privacy and measurement limits

- Restrict performance runs to the committed synthetic fixtures. The
  `PipelineTimings` record contains numeric durations only; `TiktagOutput` still
  contains original text in its in-memory replacement/map fields. Never log
  fixture text, token strings, detected entity text, `Replacement.original`, or
  reversible placeholder-map values. The runtime no longer emits token-level
  debug logs, and the harness does not initialize a logger or persist input,
  mapped values, or masked output text. It persists span/family/placeholder
  fields, per-value SHA-256 digests, a masked-text SHA-256 digest, and numeric
  confidence scores. Compare semantic fields separately from confidence.
- The harness's `peak_rss_bytes` is whole-child high-water resident memory from
  `/usr/bin/time` (macOS bytes; Linux KiB converted to bytes). It includes the
  post-timing fixture checks and an additional tokenizer loaded by the harness;
  it is not library-only attribution. `ps` snapshots are current RSS, not peak.
  CoreML may use an out-of-process service whose memory is excluded. Report the
  measurement boundary with every memory result.
- File caches are not flushed: “cold” means a new TikTag process/session, not a
  cold disk or machine. External load, power, and thermal conditions still
  affect results. Linux portability is unverified until run there.

## Older preliminary artifacts

`performance/results/baseline.json` and `performance/results/ort-profile.json`
use the pre-audit result schema and remain preliminary. The accepted artifacts
are `performance/results/validated-baseline.json` (18 rows across six fixtures
and three sessions) and `performance/results/validated-trace.json` (one separate
`multi_window` trace). Keep the older files for history; use the validated files
for baseline findings and the optimization review.

The older trace also reported mixed CPU/CoreML node placement and high CPU
`MatMulInteger`, `MatMul`, and `Where` totals. Those observations came from one
pre-audit trace and are not general provider or model claims. Interpret the
validated trace with the same limits: placement describes Node events in that
trace, and accumulated operator durations are not TikTag wall time.
