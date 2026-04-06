# lil-inference

CLI for testing quantized ONNX token-classification models.

## Workflow

1. Define a profile in `models/profiles.toml`
2. Download the model with `just download-profile <name>`
3. Run inference with `just run ... <name>`

Use `run` as the default path. `run-json` and `run-tokens` are helper commands for tooling and debugging.

## Prerequisites

- Rust toolchain
- `just`
- `hf` CLI, for downloading models

## Profile Contract

Profiles live in `models/profiles.toml`.

Each profile must define:

- `hf_repo`
- `model_dir`
- `max_tokens`
- `decode_strategy`

`model_dir` is resolved relative to the profile file directory.

Each local model bundle must contain:

- `tokenizer.json`
- `config.json`
- `onnx/model_quantized.onnx`

Model directories under `models/` are local developer assets. They are ignored by git. `models/profiles.toml` stays tracked.

## Current Profiles

### `eu_pii`

- Hugging Face: `bardsai/eu-pii-anonimization-multilang`
- Decode strategy: `pii_relaxed`
- Max tokens: `512`

### `xenova_ner_hrl`

- Hugging Face: `Xenova/distilbert-base-multilingual-cased-ner-hrl`
- Decode strategy: `generic_bio`
- Max tokens: `512`

## How to use

Build:

- `just build`

Download:

- `just download-default`
- `just download-profile eu_pii`
- `just download-profile xenova_ner_hrl`

Run:

- `just run "Contact Maria at maria@example.com"`
- `just run "John Doe lives in Paris" xenova_ner_hrl`

Helpers:

- `just sample`
- `just run-json "John Doe lives in Paris" xenova_ner_hrl`
- `just run-tokens "John Doe lives in Paris" xenova_ner_hrl`

`run-json` is for machine-readable output. `run-tokens` is for token-level debugging.

Test:

- `just test`

## Example Profile

```toml
[profiles.some_model]
hf_repo = "org/model-name"
model_dir = "some-model"
max_tokens = 512
decode_strategy = "generic_bio"
```

Then:

```bash
just download-profile some_model
just run "John Doe lives in Paris" some_model
```

## Current Limits

- ONNX-only inference
- Strict quantized ONNX bundle contract
- No chunking yet
- If tokenized input exceeds `max_tokens`, the CLI fails with a clear error
