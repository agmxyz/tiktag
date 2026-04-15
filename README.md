# tiktag

Rust library and CLI for anonymizing text with `Xenova/distilbert-base-multilingual-cased-ner-hrl`.

`tiktag` is a text-in / text-out anonymization node.

## Rust API

Use the library for Tauri, backend services, and other Rust binaries.

```rust
use std::path::Path;
use tiktag::Tiktag;

let mut tiktag = Tiktag::new(Path::new("models/profiles.toml"))?;
let result = tiktag.anonymize("Maria Garcia from OpenAI visited Berlin.")?;

assert_eq!(
    result.anonymization.anonymized_text,
    "[PERSON_1] from [ORG_1] visited [LOCATION_1]."
);
```

`Tiktag::new` loads the built-in profile and model bundle once.
`Tiktag::anonymize` is blocking and reuses the loaded runtime.
Pass an explicit `profiles.toml` path. The library does not resolve model assets from cwd on its own.

## CLI

Use the CLI for shell pipelines and one-shot runs.

- `tiktag "Maria Garcia from OpenAI visited Berlin."`
- `echo "Contact Maria at maria@example.com" | tiktag --stdin`
- `echo "Contact Maria at maria@example.com" | tiktag --stdin --json`
- `echo "Contact Maria at maria@example.com" | tiktag --stdin --debug-json`

`--json` is safe machine-readable output.
`--debug-json` includes reversible metadata for local debugging.
`--show-tokens` prints token predictions to stderr.

## Dev

- Rust toolchain
- `just`
- `hf` CLI for downloading model assets
- `just download`
- `cargo run -- download`
- `just build`
- `just test`
- `just clippy`
- `just test-fixtures`
- `just package`
- `just smoke-package`

Packaged artifact runs from `dist/tiktag` with no Rust toolchain and no `hf` CLI.

## JSON Contract

`--json` is the stable machine-readable boundary. It is safe by default and does not include reversible replacement metadata.

Current fields:

- `schema_version`
- `provenance.app_version`
- `provenance.hf_repo`
- `provenance.bundle_sha256`
- `profile`
- `anonymized_text`
- `stats`

Breaking JSON shape changes require a `schema_version` bump. Additive fields may be introduced within the same schema version.

## Internal Config

The built-in profile lives in `models/profiles.toml`.
Neither the library nor the CLI exposes model selection or profile overrides.

The config keeps the historical TOML shape:

- `default_profile`
- `[profiles.distilbert_ner_hrl]`
- `hf_repo`
- `model_dir`
- `max_tokens`
- `overlap_tokens`

`model_dir` is resolved relative to the config file directory.

Required model bundle files:

- `tokenizer.json`
- `config.json`
- `onnx/model_quantized.onnx`
