# Performance experiment correctness contract

This document separates independent fixture labels from behavior-preservation
checks for a later optimization. It does not authorize model, decoding, windowing,
or masking changes.

## Independent labels and validated output quality

The six fixture `expected` lists use UTF-8 byte offsets with exclusive ends. The
label audit sliced the encoded fixture bytes, checked both boundaries, and
reviewed the surrounding synthetic text and family. A non-ignored Rust test keeps
each range valid, sorted, non-overlapping, and in the supported family set.
These labels remain fixed when model output disagrees.

| Fixture | Input tokens / windows | Exact labels | Misses | Wrong-family outputs | Partial intervals | Spurious outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `short` | 10 / 1 | 3/3 | 0 | 0 | 0 | 0 |
| `near_limit` | 503 / 1 | 2/2 | 0 | 0 | 0 | 0 |
| `boundary` | 616 / 2 | 2/2 | 0 | 0 | 0 | 0 |
| `multi_window` | 1,586 / 4 | 17/24 | 7 | 7 | 0 | 0 |
| `unicode_repeated` | 28 / 1 | 3/5 | 2 | 0 | 0 | 0 |
| `no_entity` | 107 / 1 | 0/0 | 0 | 0 | 0 | 0 |

The accepted `performance/results/validated-baseline.json` records the actual
tokenizer boundary with `max_tokens=512` and `overlap_tokens=128`: the first
window ends at byte 2041. Expected `PERSON` `[2036, 2048)` (`Alice Morgan`)
crosses it. The `boundary` fixture has two windows and expected `LOCATION` `Paris`
at `[2057, 2062)`.

For `multi_window`, same-offset outputs mask the expected bytes under the wrong
family: `[760,772)`, `[1562,1574)`, `[2364,2376)`, `[3166,3178)`, `[5572,5584)`,
and `[6374,6386)` are expected `PERSON` but emitted `ORG`; `[3990,3999)` is
expected `ORG` but emitted `LOCATION`. Each wrong-family output counts as one
missed label and one incorrect output. `unicode_repeated` misses the two expected
`PERSON` mentions `[0,14)` and `[47,61)` (`Élodie Martin`). Other expected Unicode
labels are `José García`, `Zürich`, and `elodie@example.com`.

These counts come from `performance/results/validated-baseline.json`: three rows
per fixture, with the same quality arrays in each row. The old
`performance/results/baseline.json` remains draft. The validated result confirms
the same quality counts and semantic outputs as the draft. All 18 validated rows
match `behavior_spans.json` replacement tuples, placeholder-map digests, masked
text digests, and window counts. The expected labels and this pinned synthetic
set do not estimate production accuracy.

## Behavior-preservation checks

Keep output preservation separate from label quality. For matching model,
tokenizer, profile, and provider, require:

- exact accepted `(start, end, family, placeholder)` tuples;
- the exact placeholder-to-original mapping, including reuse for repeated values;
- exact masked text, valid UTF-8 byte ranges, sorted non-overlapping replacements,
  and no duplicate replacements;
- expected window counts and an unchanged short call after each long call on the
  same `Tiktag` instance.

`performance/fixtures/behavior_spans.json` pins replacement tuples and placeholder
IDs, a SHA-256 digest per replacement original, a SHA-256 digest per
placeholder-map value, a SHA-256 digest of the complete masked text, and the
window count. It stores no original strings, raw map values, or masked output
text. The test checks map values against replacement originals in memory, verifies
repeated-value identity, and compares the digests. The span arrays were checked
against draft result rows; placeholder IDs and digests are derived from those
spans and the current masking rules. The ignored release model check passed on
local assets whose profile, config, tokenizer, and quantized ONNX hashes match the
validated run. The old `baseline.json` remains draft; `validated-baseline.json`
confirms the same quality and semantic counts.

Confidence is outside semantic equality. Keep scores in separate numeric fields
and report any score drift separately from span, identity, and masked-text
equality. Investigate confidence drift before accepting an optimization.

The model-dependent check is ignored by default. After downloading local assets,
run:

```sh
cargo run --locked --release -- download
cargo test --locked --release fixture_regression_synthetic_pipeline -- --ignored
```

When assets are absent, the explicit test reports missing paths and the download
command. A current upstream download with different model, tokenizer, profile, or
ONNX hashes is a smoke test, not comparable evidence. The benchmark confirms the
boundary using the actual tokenizer and configured truncation/stride; it does not
infer the boundary from filler character counts. See `performance/README.md` and
`docs/performance-profiling.md` for run commands and evidence limits.

## Agreed experiment gates

The primary workload is a reused-library, four-window call. Keep initialization,
first-call, and short-call results as guardrails. Compare the same six fixtures,
model/tokenizer/profile hashes, build settings, thread settings, provider, and
machine. On the Apple M2 Pro host, collect three fresh sessions per variant; each
session uses five warm-ups and 30 warm samples per fixture. Report pooled median
and p95 with sample counts, plus each session's median. Confirm the result with a
rerun that reverses baseline/candidate order.

Retain an optimization only when the pooled multi-window median improves by at
least 5% and all three session comparisons improve; no other fixture has a
reproducible median regression above 5%; and behavior preservation passes.
Investigate any p95, initialization, or peak-RSS regression above 10% and repeat
it because tails and whole-process RSS are noisy. Do not accept an unexplained,
reproducible regression beyond those limits. These are practical thresholds, not
statistical-significance claims. If evidence stays ambiguous, retain no runtime
optimization.

Baseline p95 varies materially across sessions. Peak RSS covers the whole harness,
including its post-timing tokenizer; CoreML service memory may be outside process
RSS. Treat those numbers as comparative observations, not allocation attribution.
Performance conclusions apply to the recorded Mac; Linux is unverified.
