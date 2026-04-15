# tiktag

Rust lib + CLI for text anonymization with built-in `Xenova/distilbert-base-multilingual-cased-ner-hrl`.

## Library quickstart

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

## CLI quickstart

```bash
tiktag "Maria Garcia from OpenAI visited Berlin."
echo "Contact Maria at maria@example.com" | tiktag --stdin
echo "Contact Maria at maria@example.com" | tiktag --stdin --json
echo "Contact Maria at maria@example.com" | tiktag --stdin --debug-json
```

Flags:
- `--json` safe machine output.
- `--debug-json` reversible metadata (`replacements`, `placeholder_map`).
- `--show-tokens` token predictions to stderr.

## JSON contract

`--json` stable + safe. Fields: `schema_version`, `provenance`, `profile`, `anonymized_text`, `stats`.
Breaking shape change => bump `schema_version`.

## Dev

```bash
just download
just test
just smoke-package
```

## Built-in profile

Fixed config path: `models/profiles.toml`. No model/profile override flags.
`model_dir` resolves relative to config directory.
Required files: `tokenizer.json`, `config.json`, `onnx/model_quantized.onnx`.
