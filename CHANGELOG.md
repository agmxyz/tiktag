# Changelog

## [0.1.2] - 2026-04-23

### Fixed
- `tiktag download` now bootstraps missing profile file for fresh installs.
- Installed binary now stores and resolves model profile/assets from OS app-data path, so runs work from arbitrary working directories.

## [0.1.1] - 2026-04-23

### Added
- MIT licensing + security policy + CI workflow for OSS release baseline.
- Crates.io metadata and package include-list hardening.
- README install/quickstart, light caveat, and model attribution updates.


## [0.1.0] - 2026-04-22

### Added
- Rust library + CLI for multilingual text anonymization.
- Built-in profile flow using `Xenova/distilbert-base-multilingual-cased-ner-hrl` quantized ONNX assets.
- CLI modes: text output, `--json`, `--debug-json`, and `download`.
- Sliding-window inference support with stable placeholder mapping per call.
- Benchmarks and fixture-based regression tests.
