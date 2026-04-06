# tiktag

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
- `overlap_tokens`
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
- `just test-fixtures`

Fixture tests in `testdocs/` are profile-aligned and intentionally partial: they assert stable repeated entities for each model profile and verify long inputs use multiple windows.

## Example Profile

```toml
[profiles.some_model]
hf_repo = "org/model-name"
model_dir = "some-model"
max_tokens = 512
overlap_tokens = 128
decode_strategy = "generic_bio"
```

Then:

```bash
just download-profile some_model
just run "John Doe lives in Paris" some_model
```

## Adding a new model

### If the model fits an existing decode strategy

If the model uses standard BIO tagging (`B-PER`, `I-PER`, ...), use `generic_bio`. If it matches the eu-pii label set, use `pii_relaxed`.

1. Add a profile to `models/profiles.toml`
2. `just download-profile <name>`
3. `just run "test input" <name>`

No code changes needed.

### If the model needs a new decode strategy

The merge logic that combines token-level predictions into entity spans is model-specific. It depends on the tagging scheme (BIO, BIOES, bare labels), gap tolerance between tokens, and any misprediction patterns worth compensating for. These aren't declared by the model — you discover them by inspecting the label set and testing.

All changes live in `src/decode.rs`:

1. Add a variant to the `DecodeStrategy` enum
2. Write a `merged_label_<name>` function (see `merged_label_generic` and `merged_label_pii_relaxed` as examples)
3. Add the new arm to the `merged_label` match
4. Add the variant to the `Display` impl
5. Add tests for the new merge behavior
6. Add a profile to `models/profiles.toml` referencing the new strategy
7. `just test`

Use `just run-tokens "..." <name>` to inspect token-level predictions while developing merge rules.

## Current Limits

- ONNX-only inference
- Strict quantized ONNX bundle contract
- Sliding-window inference requires `overlap_tokens > 0`
- Fixture regression tests require local downloaded model assets
