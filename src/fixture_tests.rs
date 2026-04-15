use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use serde::Deserialize;

use crate::Tiktag;

const BUILTIN_PROFILE_NAME: &str = "distilbert_ner_hrl";
const BUILTIN_MODEL_DIR: &str = "models/distilbert-base-multilingual-cased-ner-hrl";
const REQUIRED_MODEL_FILES: &[&str] =
    &["tokenizer.json", "config.json", "onnx/model_quantized.onnx"];

#[derive(Debug, Deserialize)]
struct FixtureManifest {
    profile: String,
    min_window_count: usize,
    #[serde(default)]
    expected_replacements: Vec<ExpectedReplacement>,
    #[serde(default)]
    forbidden_literals: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ExpectedReplacement {
    placeholder: String,
    original: String,
    count: usize,
}

fn project_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn load_manifest(base_name: &str) -> anyhow::Result<FixtureManifest> {
    let manifest_path = project_path(&format!("testdocs/{base_name}_expected.toml"));
    let manifest_text = fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    toml::from_str(&manifest_text)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))
}

fn load_input(base_name: &str) -> anyhow::Result<String> {
    let input_path = project_path(&format!("testdocs/{base_name}_input.md"));
    fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))
}

fn require_local_model_assets() -> anyhow::Result<()> {
    let model_dir = project_path(BUILTIN_MODEL_DIR);
    let missing = REQUIRED_MODEL_FILES
        .iter()
        .map(|relative_path| model_dir.join(relative_path))
        .filter(|path| !path.is_file())
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();

    if missing.is_empty() {
        return Ok(());
    }

    bail!(
        "fixture tests for profile '{BUILTIN_PROFILE_NAME}' require local model assets; missing: {}. Run `just download` first.",
        missing.join(", ")
    );
}

fn run_fixture(base_name: &str) -> anyhow::Result<()> {
    let manifest = load_manifest(base_name)?;
    let input = load_input(base_name)?;

    assert_eq!(
        manifest.profile, BUILTIN_PROFILE_NAME,
        "fixture '{base_name}' must target the built-in profile"
    );

    require_local_model_assets()?;

    let mut tiktag = Tiktag::new(&project_path("models/profiles.toml"))?;
    let result = tiktag.anonymize(&input)?;

    assert!(
        result.window_count >= manifest.min_window_count,
        "fixture '{}' expected at least {} windows, got {}",
        base_name,
        manifest.min_window_count,
        result.window_count
    );

    for expected in manifest.expected_replacements {
        let replacement_count = result
            .anonymization
            .replacements
            .iter()
            .filter(|replacement| replacement.placeholder == expected.placeholder)
            .count();
        assert_eq!(
            replacement_count, expected.count,
            "fixture '{}' expected {} replacement(s) for placeholder {}, got {}",
            base_name, expected.count, expected.placeholder, replacement_count
        );
        assert_eq!(
            result
                .anonymization
                .placeholder_map
                .get(&expected.placeholder)
                .map(String::as_str),
            Some(expected.original.as_str()),
            "fixture '{}' expected placeholder {} to map to {}",
            base_name,
            expected.placeholder,
            expected.original
        );
        assert!(
            result
                .anonymization
                .anonymized_text
                .contains(&expected.placeholder),
            "fixture '{}' expected anonymized text to contain {}",
            base_name,
            expected.placeholder
        );
        assert!(
            !result
                .anonymization
                .anonymized_text
                .contains(&expected.original),
            "fixture '{}' expected anonymized text to remove {}",
            base_name,
            expected.original
        );
    }

    for forbidden_literal in manifest.forbidden_literals {
        assert!(
            !result
                .anonymization
                .anonymized_text
                .contains(&forbidden_literal),
            "fixture '{base_name}' expected anonymized text to remove forbidden literal {forbidden_literal}"
        );
    }

    Ok(())
}

#[test]
#[ignore = "requires local downloaded model assets; run `just test-fixtures`"]
fn fixture_regression_xenova_ner_windowed() -> anyhow::Result<()> {
    run_fixture("xenova_ner_windowed")
}

#[test]
#[ignore = "requires local downloaded model assets; run `just test-fixtures`"]
fn fixture_regression_xenova_ner_stress_windowed() -> anyhow::Result<()> {
    run_fixture("xenova_ner_stress_windowed")
}
