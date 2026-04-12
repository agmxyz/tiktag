use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use serde::Deserialize;

use crate::anonymize;
use crate::decode::EntitySpan;
use crate::model_bundle::missing_model_files;
use crate::profiles::{BUILTIN_PROFILE_NAME, Profiles};
use crate::runtime::{LoadResult, ModelRuntime};

#[derive(Debug, Deserialize)]
struct FixtureManifest {
    profile: String,
    min_window_count: usize,
    #[serde(default)]
    expected: Vec<ExpectedEntity>,
    #[serde(default)]
    expected_replacements: Vec<ExpectedReplacement>,
    #[serde(default)]
    forbidden_literals: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ExpectedEntity {
    label: String,
    text: String,
    count: usize,
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

fn require_local_model_assets(model_dir: &Path, profile_name: &str) -> anyhow::Result<()> {
    let missing = missing_model_files(model_dir);

    if missing.is_empty() {
        return Ok(());
    }

    let missing_list = missing
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");

    bail!(
        "fixture tests for profile '{profile_name}' require local model assets; missing: {missing_list}. Run `just download` first."
    );
}

fn exact_entity_counts(entities: &[EntitySpan]) -> BTreeMap<(String, String), usize> {
    let mut counts = BTreeMap::new();
    for entity in entities {
        *counts
            .entry((entity.label.clone(), entity.text.clone()))
            .or_default() += 1;
    }
    counts
}

fn assert_no_exact_duplicate_spans(base_name: &str, entities: &[EntitySpan]) {
    let mut seen = BTreeSet::new();

    for entity in entities {
        let key = (
            entity.label.clone(),
            entity.start,
            entity.end,
            entity.text.clone(),
        );
        assert!(
            seen.insert(key),
            "fixture '{base_name}' produced an exact duplicate span: {entity:?}"
        );
    }
}

fn run_fixture(base_name: &str) -> anyhow::Result<()> {
    let manifest = load_manifest(base_name)?;
    let input = load_input(base_name)?;

    assert_eq!(
        manifest.profile, BUILTIN_PROFILE_NAME,
        "fixture '{base_name}' must target the built-in eu_pii profile"
    );

    let profiles = Profiles::load(&project_path("models/profiles.toml"))?;
    let resolved = profiles.resolve_default();
    require_local_model_assets(&resolved.model_dir, &manifest.profile)?;

    let LoadResult { mut runtime, .. } = ModelRuntime::load(&resolved)?;
    let result = runtime.infer(&input, false)?;

    assert!(
        result.window_count >= manifest.min_window_count,
        "fixture '{}' expected at least {} windows, got {}",
        base_name,
        manifest.min_window_count,
        result.window_count
    );

    assert_no_exact_duplicate_spans(base_name, &result.entities);
    assert!(
        result.timings.total_ms > 0.0,
        "fixture '{base_name}' expected infer.total_ms to be positive"
    );

    let actual_counts = exact_entity_counts(&result.entities);
    for expected in manifest.expected {
        let key = (expected.label.clone(), expected.text.clone());
        let actual = actual_counts.get(&key).copied().unwrap_or(0);
        assert_eq!(
            actual, expected.count,
            "fixture '{}' expected {} occurrence(s) of [{}] {}, got {}",
            base_name, expected.count, expected.label, expected.text, actual
        );
    }

    if !manifest.expected_replacements.is_empty() || !manifest.forbidden_literals.is_empty() {
        let anonymized = anonymize::anonymize(&input, &result.entities)?;
        for expected in manifest.expected_replacements {
            let replacement_count = anonymized
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
                anonymized
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
                anonymized.anonymized_text.contains(&expected.placeholder),
                "fixture '{}' expected anonymized text to contain {}",
                base_name,
                expected.placeholder
            );
            assert!(
                !anonymized.anonymized_text.contains(&expected.original),
                "fixture '{}' expected anonymized text to remove {}",
                base_name,
                expected.original
            );
        }

        for forbidden_literal in manifest.forbidden_literals {
            assert!(
                !anonymized.anonymized_text.contains(&forbidden_literal),
                "fixture '{}' expected anonymized text to remove forbidden literal {}",
                base_name,
                forbidden_literal
            );
        }
    }

    Ok(())
}

#[test]
#[ignore = "requires local downloaded model assets; run `just test-fixtures`"]
fn fixture_regression_eu_pii_windowed() -> anyhow::Result<()> {
    run_fixture("eu_pii_windowed")
}

#[test]
#[ignore = "requires local downloaded model assets; run `just test-fixtures`"]
fn fixture_regression_eu_pii_stress_windowed() -> anyhow::Result<()> {
    run_fixture("eu_pii_stress_windowed")
}
