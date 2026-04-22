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

Keep only contract + invariants + real footguns. Replace stale text; no history log.

## Project

`tiktag` = text anonymizer.
- One Rust crate: library + thin CLI.
- Built-in model: `Xenova/distilbert-base-multilingual-cased-ner-hrl` (quantized ONNX).
- `AGENTS.md` = authoritative implementation contract. `README.md` = user summary.

## Library contract

```rust
use tiktag::{Tiktag, TiktagError, TiktagOutput};

let mut tiktag = Tiktag::new(&profiles_path)?;   // loads tokenizer + ONNX session (~350 ms)
let out: TiktagOutput = tiktag.anonymize(text)?; // per-call, ms-to-tens-of-ms
let text = &out.anonymization.anonymized_text;
```

- Construct once, call many: `new` expensive; `anonymize` reuses runtime.
- `anonymize` takes `&mut self`. No internal locking. Multi-thread hosts use `Mutex<Tiktag>` or per-thread instance.
- `profiles_path` explicit. Relative `model_dir` resolves against profile file parent only.
- `TiktagOutput`: `anonymization` + `sequence_len` + `window_count`.
- Errors are `TiktagError` (`thiserror`). Keep typed variants for boundary/inference/profile failures. `Other(anyhow::Error)` is escape hatch.
- Placeholder numbering stable per call only. No cross-document identity.

## CLI contract

- `tiktag "<text>"` or `tiktag --stdin` produces anonymized text on stdout (`println!`, trailing newline).
- `tiktag --json` emits machine-readable output without reversible metadata.
- `tiktag --debug-json` emits the reversible map; local/debug use only.
- `tiktag download` fetches the bundled model.
- Diagnostics/timing logs go to stderr via `log`; JSON modes also emit `stats.timings` on stdout payload.
- The CLI resolves `models/profiles.toml` from app-data (`.../tiktag/models/profiles.toml`) first, then legacy fallback (`<exe_dir>/models/profiles.toml`, then cwd). No model/profile selection flags.
- Prefer `--stdin` for large inputs: avoids shell argv limits and composes with pipelines.

## JSON output

- Fields: `schema_version`, `provenance`, `profile`, `anonymized_text`, `stats`.
- `stats.timings` is machine-dependent — content-hash pipelines must ignore it.
- Additive field changes keep `schema_version`. Removing, renaming, or retyping a field bumps it.

## Caveats

- macOS builds register CoreML EP; non-macOS uses CPU EP.
- ORT may silently fall back to CPU if CoreML fails.
- CoreML compile not cached to disk: CLI pays compile each invocation; long-lived library host pays once per process.
