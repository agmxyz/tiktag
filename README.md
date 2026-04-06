# lil-inference

Pragmatic CLI for testing quantized ONNX token-classification models in Rust.

## Quick Start

1. Build:
   - `just build`
2. Restore the default model:
   - `just download-default`
3. Run the default profile:
   - `just sample`
   - `just run "Contact Maria at maria@example.com"`

## Profiles

Profiles live in `models/profiles.toml`.

- `default_profile` selects the default model.
- Each profile defines:
  - `hf_repo`
  - `model_dir`
  - `max_tokens`
  - `decode_strategy`

`model_dir` is resolved relative to the profile file directory.

Required files per model bundle:

- `tokenizer.json`
- `config.json`
- `onnx/model_quantized.onnx`

Model directories under `models/` are local developer assets. They are ignored by git. `models/profiles.toml` stays tracked.

## Commands

- Run human output:
  - `just run "John Doe lives in Paris"`
  - `just run "John Doe lives in Paris" xenova_ner_hrl`
- Run JSON output:
  - `just run-json "John Doe lives in Paris" xenova_ner_hrl`
- Show token debug output:
  - `just run-tokens "John Doe lives in Paris" xenova_ner_hrl`
- Run tests:
  - `just test`

## Model Restore

Restore a named profile from Hugging Face:

- `just download-profile eu_pii`
- `just download-profile xenova_ner_hrl`

The download command reads `hf_repo` and `model_dir` from `models/profiles.toml`. Keep profiles as the single source of truth.

## Find Compatible Models

Search for repos that match the current runtime contract:

- `just search-supported`
- `just search-supported "multilingual ner onnx" 200`

Current filter keeps token-classification models with:

- `config.json`
- `tokenizer.json`
- `onnx/model_quantized.onnx`

## Current Limits

- ONNX-only inference.
- Quantized ONNX bundle contract is strict by design.
- If tokenized input exceeds `max_tokens`, the CLI fails with a clear error.
- Sliding-window chunking is deferred to a later iteration.
