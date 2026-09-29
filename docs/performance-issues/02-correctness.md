Spec for [GitHub issue #23](https://github.com/agmxyz/tiktag/issues/23), child of
[#21](https://github.com/agmxyz/tiktag/issues/21).

## Goal

Validate independent fixture labels and pin current masking behavior without
changing model, decoding, window stitching, or anonymization behavior. Keep
model-quality evaluation separate from before/after behavior preservation.

## Owned files

- `performance/fixtures/**`
- `src/fixture_tests.rs`
- `docs/performance-correctness.md`

Coordinate the fixture schema with #22. Do not edit runtime, harness/driver,
`Cargo.toml`, root plan/issue docs, or optimize inference.

## Non-goals

Do not relabel expected entities merely to match current model output. Do not fix
model quality, change masking semantics, or run the performance baseline. Do not
claim production anonymization accuracy from these synthetic examples.

## Existing draft observations

Keep labels as independent expected values and report baseline output explicitly:

- `short`: 3/3 exact; `near_limit`: 2/2 at 503/512 tokens.
- `boundary`: 2/2 exact; tokenizer first-window end is byte 2041 and person
  span `[2036, 2048)` crosses it.
- `multi_window`: 17/24 exact. Seven spans are masked at the expected bytes under
  the wrong family; count each as a missed expected label and an incorrect-family
  output, not as unmasked text.
- `unicode_repeated`: 3/5 exact; expected accented-name spans `[0, 14)` and
  `[47, 61)` are missed.
- `no_entity`: 0/0 exact and no outputs.

These are pinned fixture observations, not accuracy estimates. Details are in
`docs/performance-correctness.md`.

## Acceptance

- Independently verify each expected label's UTF-8 byte offsets, exclusive end,
  and family. Confirm the boundary with the actual tokenizer and profile stride.
- Report exact matches, missed labels, partial intervals, wrong-family masks and
  spurious outputs separately. A same-offset wrong-family mask is both an exact
  miss and an incorrect output.
- Pin behavior preservation separately: exact span/family/placeholder ID tuples,
  placeholder-to-original mapping (including repeated identity), and exact masked
  text. Keep confidence scores outside semantic equality and report any drift.
- Check UTF-8 slice validity, sorted/non-overlapping and non-duplicate spans,
  window counts, repeated-value identity, and a short call after long calls on the
  same `Tiktag` instance.
- Model-dependent checks stay ignored by default and explain missing local assets
  when explicitly invoked. Preserve independently labeled misses in the report.

## Verification commands

From repository root:

```sh
cargo test --locked
cargo test --locked --release fixture_regression_synthetic_pipeline -- --ignored
cargo test --locked --release fixture_regression -- --ignored
```

The ignored checks require the exact model bundle. If a download is needed, obtain
it with `cargo run --locked --release -- download`, then verify profile, config,
tokenizer and ONNX file hashes against the run being reproduced. A current upstream
download with different hashes is only a smoke test, not a comparable run. Do not
run baseline benchmarks here; root schedules #24 after this and #22 complete.

## Dependency and handoff

No blockers; may run in parallel with #22. #24 is natively blocked by #22 and #23.
Return changed files, commands/results, label audit, expected-quality counts,
observed behavior snapshot fields, and any remaining gap.
