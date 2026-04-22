# Changelog


## [0.1.0] - 2026-04-22

### Added
- Rust library + CLI for multilingual text anonymization.
- Built-in profile flow using `Xenova/distilbert-base-multilingual-cased-ner-hrl` quantized ONNX assets.
- CLI modes: text output, `--json`, `--debug-json`, and `download`.
- Sliding-window inference support with stable placeholder mapping per call.
- Benchmarks and fixture-based regression tests.
