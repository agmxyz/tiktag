# AGENTS.md

## Living document policy

This file is a living project memory for future sessions.

Add only high-signal information:

- critical design decisions and invariants
- model-specific behavior and known quirks
- proven defaults and operational shortcuts
- recurring footguns and how to avoid them

Keep entries concise, practical, and easy to scan.
Remove or replace stale guidance when it is no longer true.

## Project intent

This experiment is the first iteration of a developer tool for testing PII and NER models.

The tool exists to let developers quickly validate model behavior on real text inputs:

- load a model
- run inference
- inspect outputs and timings
- debug obvious errors fast

## Key design decisions

### Product direction

- Prefer pragmatic, boring solutions.
- Prefer low cognitive load over flexible abstractions.
- Keep the first iteration CLI-first and easy to run locally.
- Optimize for "easy to understand in one read" instead of "future perfect architecture".

### What to prioritize

- Deterministic behavior and clear logs.
- Simple defaults that work without extra config.
- Small, incremental changes that are easy to review.
- Fast feedback loops for developers testing models.

### What to avoid (until needed)

- Complex plugin systems.
- Heavy framework layers.
- Premature generalization for model orchestration.
- Clever abstractions that hide control flow.

### Runtime and model assumptions (v1)

- Inference is ONNX-only in this iteration.
- Models are selected by profile from `models/profiles.toml`.
- Profile entries must include `hf_repo`, `model_dir`, `max_tokens`, and `decode_strategy`. All four are required at parse time — do not add serde defaults.
- Profile-driven runs are the only supported workflow. Do not add ad hoc `--model-dir` overrides back in.
- If tokenized input exceeds profile `max_tokens` (typically 512), fail with a clear error. Note that `max_tokens` includes special tokens ([CLS]/[SEP]), so effective user-text capacity is `max_tokens - 2`.
- `config.json` must have a contiguous `id2label` map (no gaps). The loader rejects sparse maps at startup.
- Sliding-window chunking is explicitly deferred to a later iteration.
- The runtime contract is intentionally strict: `tokenizer.json`, `config.json`, and `onnx/model_quantized.onnx`.
- Local model directories under `models/` are disposable developer assets and should stay ignored by git.
- Keep the secondary profile name aligned to the source model (`xenova_ner_hrl`) to avoid alias confusion.
- Restore local assets with `just download-default` or `just download-profile <name>`.
- The core user workflow is: define a profile, `just download-profile <name>`, then run inference.
- `hf` CLI is a download prerequisite only. It is not part of runtime inference.
- Keep `just run` as the default documented path. Treat `run-json` and `run-tokens` as helper recipes for tooling and debugging.

### Decode strategy behavior

- `generic_bio`: Standard BIO scheme. Merges consecutive tokens only when the previous is B-X and the next is I-X (same entity kind). Requires an explicit I-prefix for continuation.
- `pii_relaxed`: Relaxed merging for the eu-pii model. Ignores BIO prefixes for continuation decisions. Allows cross-token gap merging for emails (`.@_-+`), phones (whitespace, `+-()./`), and whitespace gaps for other types. Includes a model-specific heuristic: absorbs ORGANIZATION_NAME tokens into a preceding EMAIL_ADDRESS span (the eu-pii model frequently classifies email domains as org names).
- Some models emit bare labels (e.g. `EMAIL_ADDRESS`) without BIO prefixes. `split_label` treats these as B-tags.
- Decode strategies are intentionally not generalizable. Each model family may have different tagging schemes, gap tolerances, and misprediction patterns. Adding a new model that doesn't fit an existing strategy requires a new `DecodeStrategy` variant and merge function in `decode.rs`. This is by design — a "general" solution would hide model-specific judgment calls.

### Logging and output conventions

- Use native Rust logging (`log` + `env_logger`), not ad-hoc `eprintln!` for runtime diagnostics.
- Keep model predictions on stdout and diagnostics/timings on stderr via logger.
- Support both human-readable output and JSON output for developer tooling.

## Reviewing and writing code

- Use the justfile
- Treat this file as shared memory across sessions; update it when new relevant project knowledge is discovered.
