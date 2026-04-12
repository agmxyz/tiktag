# tiktag

CLI for anonymizing text with `bardsai/eu-pii-anonimization-multilang`.

`tiktag` is a text-in / text-out anonymization node. It does not parse PDFs or rebuild them. The intended pipeline boundary is: another tool extracts text, `tiktag` anonymizes it, and another tool handles document reconstruction.

## Workflow

1. For local source checkouts, download the built-in model with `just download` or `tiktag download`
2. Build a self-contained artifact with `just package` when distributing or deploying
3. Run anonymization on literal text or pipe text with `--stdin`
4. Use `--json` only when you need machine-readable replacement metadata and total timings

`run` is the default path. `run-json` and `run-tokens` are helper commands for tooling and debugging.

## Prerequisites

For development from source:

- Rust toolchain
- `just`
- `hf` CLI for downloading model assets

For running a packaged artifact:

- no Rust toolchain
- no `hf` CLI

## Commands

Build and test:

- `just build`
- `just test`
- `just clippy`
- `just test-fixtures`

Download model assets:

- `just download`
- `cargo run -- download`

Create and verify a packaged artifact:

- `just package`
- `just smoke-package`

Run from packaged artifact root (`dist/tiktag`):

- `echo "Contact Maria at maria@example.com" | ./dist/tiktag/tiktag --stdin`
- `echo "Contact Maria at maria@example.com" | ./dist/tiktag/tiktag --stdin --json`

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

Model bundle must contain:

- `tokenizer.json`
- `config.json`
- `onnx/model_quantized.onnx`
