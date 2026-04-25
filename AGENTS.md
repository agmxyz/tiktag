# AGENTS.md

Authoritative project contract for `tiktag`. Keep this file to contract, invariants, and real caveats only. `README.md` stays user-facing and short.

## Working rules

- state assumptions when they matter; if ambiguity changes implementation, ask instead of guessing
- prefer smallest change that solves request; no speculative features, abstractions, or configurability
- touch requested scope only; do not refactor or clean unrelated code
- verify behavior-changing work with targeted checks before declaring done

## Project

`tiktag` = text anonymizer.

- one Rust crate: library + thin CLI
- built-in model: `Xenova/distilbert-base-multilingual-cased-ner-hrl` (quantized ONNX)
- built-in model scope: `PERSON`, `ORG`, `LOCATION`
- regex recognizers are additive supplements after model inference
- current built-in regex recognizer: email

## Library contract

```rust
use tiktag::{Tiktag, TiktagError, TiktagOutput};

let mut tiktag = Tiktag::new(&profiles_path)?;   // loads tokenizer + ONNX session once
let out: TiktagOutput = tiktag.anonymize(text)?; // reuses runtime per call
let text = &out.anonymization.anonymized_text;
```

- construct once, call many: `new` is expensive; `anonymize` reuses runtime
- `anonymize` takes `&mut self`; no internal locking
- multi-thread hosts use `Mutex<Tiktag>` or per-thread instance
- `profiles_path` is explicit
- relative `model_dir` resolves against profile file parent only
- `TiktagOutput` = `anonymization` + `sequence_len` + `window_count`
- errors are `TiktagError`; keep typed variants for profile, bundle, and inference boundaries
- placeholder numbering is stable per call only; no cross-document identity

## CLI contract

- `tiktag "<text>"` or `tiktag --stdin` prints anonymized text to stdout with trailing newline
- `tiktag --json` emits machine-readable output without reversible metadata
- `tiktag --debug-json` emits reversible metadata for local/debug use only
- `tiktag download` fetches bundled model assets
- diagnostics and timing logs go to stderr via `log`
- JSON modes also emit `stats.timings` on stdout payload
- CLI resolves `models/profiles.toml` from app-data first, then legacy fallback: `<exe_dir>/models/profiles.toml`, then cwd
- no model/profile selection flags
- prefer `--stdin` for large inputs

## Profiles and recognizers

- built-in config file: `models/profiles.toml`
- `model_dir` in that file resolves relative to config directory unless absolute
- recognizer toggles live under `[recognizers]`
- v1 recognizer surface: `email = true|false`, default `true`

## JSON contract

- top-level fields: `schema_version`, `provenance`, `profile`, `anonymized_text`, `stats`
- `stats.timings` is machine-dependent; hash-based pipelines must ignore it
- additive field changes keep `schema_version`
- removing, renaming, or retyping a field bumps `schema_version`

## Verification baseline

- local baseline: `cargo clippy --all-targets --all-features -- -D warnings` and `cargo test`
- fixture regressions are manual because they require local downloaded model assets

## Caveats

- macOS builds register CoreML EP; non-macOS uses CPU EP
- ORT may silently fall back to CPU if CoreML fails
- CoreML compile is not cached to disk: CLI pays compile each invocation; long-lived library host pays once per process
- fixture placeholder ids and counts are pinned to current built-in model/profile, not intended as cross-model guarantees
