# AGENTS.md

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a staff engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Living document policy

This file is shared project memory for future sessions.

Keep only high-signal information:

- current product contract and invariants
- model-specific behavior that changes implementation decisions
- proven defaults and operational shortcuts
- recurring footguns worth preventing

Keep it short. Replace stale guidance instead of accumulating history.

## Project

`tiktag` is a text anonymizer. One Rust crate ships a library and a thin CLI. Built-in model: `Xenova/distilbert-base-multilingual-cased-ner-hrl` (quantized ONNX). Authoritative contract lives here; `README.md` is a short user-facing summary of the same shape.

## Library contract

```rust
use tiktag::{Tiktag, TiktagError, TiktagOutput};

let mut tiktag = Tiktag::new(&profiles_path)?;   // loads tokenizer + ONNX session (~350 ms)
let out: TiktagOutput = tiktag.anonymize(text)?; // per-call, ms-to-tens-of-ms
let text = &out.anonymization.anonymized_text;
```

- Construct once, call many. `new` loads expensive state (tokenizer, ONNX session, labels); `anonymize` reuses it.
- `anonymize` takes `&mut self`. Multi-threaded hosts wrap in `Mutex<Tiktag>` or clone their own instance; the lib does no locking.
- `profiles_path` is explicit. Relative `model_dir` inside the TOML resolves against the profile file's parent — no cwd lookup elsewhere in the lib.
- `TiktagOutput` carries `anonymization` (text + replacements + stats) plus `sequence_len` and `window_count` for caller-side observability.
- Errors are `TiktagError` (thiserror). Boundary failures have dedicated variants (`ProfileRead`, `ProfileParse`, `ModelBundleMissing`, …). Inference-path errors currently fall through `Other(anyhow::Error)` — callers that need fine-grained handling should pattern on the boundary variants and treat `Other` as opaque.
- Placeholder numbering is stable per call given the entity set returned by inference. No cross-document identity.

## CLI contract

- `tiktag "<text>"` or `tiktag --stdin` produces anonymized text on stdout (`println!`, trailing newline).
- `tiktag --json` emits machine-readable output without reversible metadata.
- `tiktag --debug-json` emits the reversible map; local/debug use only.
- `tiktag download` fetches the bundled model.
- Diagnostics and timings go to stderr via `log`.
- The CLI loads `models/profiles.toml` **relative to the current working directory**. Run from the install root (or a directory containing `models/`). No model/profile selection flags.
- Prefer `--stdin` for large inputs: avoids shell argv limits and composes with pipelines.

## JSON output

- Fields: `schema_version`, `provenance`, `profile`, `anonymized_text`, `stats`.
- `stats.timings` is machine-dependent — content-hash pipelines must ignore it.
- Additive field changes keep `schema_version`. Removing, renaming, or retyping a field bumps it.

## Footguns & known-legacy

- macOS builds register the CoreML EP; other targets run CPU. ORT silently falls back to CPU if CoreML can't load. The CoreML compile is not cached to disk, so **CLI load pays recompile every invocation**; library hosts pay it once per process.
- `models/profiles.toml` uses a legacy `default_profile` + `[profiles.<name>]` shape that validates down to the single built-in profile. Adding a second profile will fail validation. Flattening to a single `[model]` section is a candidate cleanup.
- CLI resolves `models/` by cwd (see CLI contract). Exe-relative resolution is a candidate cleanup.
- Bench harness: `benches/anonymize.rs`, `just bench`. Skips when assets absent.
