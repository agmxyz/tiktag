# tiktag

Rust lib + CLI for text anonymization with built-in `Xenova/distilbert-base-multilingual-cased-ner-hrl`.

Principle: keep integrations and docs concise + pragmatic. No speculative features.

## Library

Use for Tauri/backend/other Rust hosts.

```rust
use std::path::Path;
use tiktag::Tiktag;

let mut tiktag = Tiktag::new(Path::new("models/profiles.toml"))?;
let out = tiktag.anonymize("Maria Garcia from OpenAI visited Berlin.")?;
println!("{}", out.anonymization.anonymized_text);
```

Contract:
- `Tiktag::new` loads profile + model once.
- `Tiktag::anonymize` is blocking, reuses runtime.
- Pass explicit `profiles.toml` path.

## CLI

Default output: anonymized text only.

```bash
tiktag "Maria Garcia from OpenAI visited Berlin."
echo "Contact Maria at maria@example.com" | tiktag --stdin
echo "Contact Maria at maria@example.com" | tiktag --stdin --json
echo "Contact Maria at maria@example.com" | tiktag --stdin --debug-json
```

Flags:
- `--stdin`: read text from stdin.
- `--json`: safe machine output.
- `--debug-json`: includes reversible metadata (`replacements`, `placeholder_map`).
- `--show-tokens`: token predictions to stderr.

## JSON Contract

`--json` fields:
- `schema_version`
- `provenance` (`app_version`, `hf_repo`, `bundle_sha256`)
- `profile`
- `anonymized_text`
- `stats`

Rules:
- breaking shape change => bump `schema_version`
- additive fields allowed within same version

## Dev

```bash
just download
just build
just test
just clippy
just test-fixtures
just package
just smoke-package
```

Packaged artifact in `dist/tiktag` runs without Rust toolchain or `hf` CLI.

## Built-in Profile

- fixed config path: `models/profiles.toml`
- no model/profile override flags in lib or CLI
- `model_dir` resolves relative to config directory
- required files:
  - `tokenizer.json`
  - `config.json`
  - `onnx/model_quantized.onnx`
