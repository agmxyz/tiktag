# tiktag

CLI for anonymizing text with `bardsai/eu-pii-anonimization-multilang`.

`tiktag` is a text-in / text-out anonymization node. It does not parse PDFs or rebuild them. The intended pipeline boundary is: another tool extracts text, `tiktag` anonymizes it, and another tool handles document reconstruction.

## Workflow

1. Download the built-in model with `just download` or `tiktag download`
2. Run anonymization on literal text or pipe text with `--stdin`
3. Use `--json` only when you need machine-readable replacement metadata and total timings

`run` is the default path. `run-json` and `run-tokens` are helper commands for tooling and debugging.

## Prerequisites

- Rust toolchain
- `just`
- `hf` CLI for downloading model assets

## Commands

Build and test:

- `just build`
- `just test`
- `just test-fixtures`

Download model assets:

- `just download`
- `cargo run -- download`

Run anonymization:

- `just run "Contact Maria at maria@example.com"`
- `just sample`
- `cat testdocs/eu_pii_windowed_input.md | cargo run -- --stdin --json`
- `cat testdocs/eu_pii_windowed_input.md | just run-stdin`
- `cat testdocs/eu_pii_windowed_input.md | just run-json-stdin`

`run-json` is for machine-readable output. `run-tokens` is for token-level debugging on stderr.

## Internal Config

The model settings live in `models/profiles.toml` at a fixed internal path. The CLI does not expose model selection or profile overrides.

The internal config keeps the historical TOML shape:

- `default_profile`
- `[profiles.eu_pii]`
- `hf_repo`
- `model_dir`
- `max_tokens`
- `overlap_tokens`

`model_dir` is resolved relative to the config file directory.

Each local model bundle must contain:

- `tokenizer.json`
- `config.json`
- `onnx/model_quantized.onnx`

Model directories under `models/` are local developer assets. They are ignored by git. `models/profiles.toml` stays tracked.

## JSON Output

`--json` emits:

- `profile`
- `anonymized_text`
- `replacements`
- `placeholder_map`
- `stats`

Nested timings live under `stats.timings.load.total_ms` and `stats.timings.infer.total_ms`.

## Current Limits

- ONNX-only inference
- Strict quantized ONNX bundle contract
- Sliding-window inference requires `overlap_tokens > 0`
- Fixture regression tests require local downloaded model assets
- Placeholder reuse is stable only within one document
