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
- Profile entries must include `hf_repo`, `model_dir`, `max_tokens`, and `decode_strategy`.
- Profile-driven runs are the only supported workflow. Do not add ad hoc `--model-dir` overrides back in.
- If tokenized input exceeds profile `max_tokens` (typically 512), fail with a clear error.
- Sliding-window chunking is explicitly deferred to a later iteration.
- The runtime contract is intentionally strict: `tokenizer.json`, `config.json`, and `onnx/model_quantized.onnx`.
- Local model directories under `models/` are disposable developer assets and should stay ignored by git.
- Keep the secondary profile name aligned to the source model (`xenova_ner_hrl`) to avoid alias confusion.
- Restore local assets with `just download-default` or `just download-profile <name>`.
- Use `just search-supported` to find Hugging Face repos that match the current runtime contract.

### Logging and output conventions

- Use native Rust logging (`log` + `env_logger`), not ad-hoc `eprintln!` for runtime diagnostics.
- Keep model predictions on stdout and diagnostics/timings on stderr via logger.
- Support both human-readable output and JSON output for developer tooling.

## Reviewing and writing code

- Use the justfile
- Treat this file as shared memory across sessions; update it when new relevant project knowledge is discovered.
