use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use serde::Deserialize;

use crate::decode::EntitySpan;
use crate::profiles::Profiles;
use crate::runtime::{LoadResult, ModelRuntime};

#[derive(Debug, Deserialize)]
struct FixtureManifest {
    profile: String,
    min_window_count: usize,
    expected: Vec<ExpectedEntity>,
}

#[derive(Debug, Deserialize)]
struct ExpectedEntity {
    label: String,
    text: String,
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
    let required = ["tokenizer.json", "config.json", "onnx/model_quantized.onnx"];
    let missing = required
        .iter()
        .map(|path| model_dir.join(path))
        .filter(|path| !path.is_file())
        .collect::<Vec<_>>();

    if missing.is_empty() {
        return Ok(());
    }

    let missing_list = missing
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");

    bail!(
        "fixture tests for profile '{}' require local model assets; missing: {}. Run `just download-profile {}` first.",
        profile_name,
        missing_list,
        profile_name
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
            "fixture '{}' produced an exact duplicate span: {:?}",
            base_name,
            entity
        );
    }
}

fn run_fixture(base_name: &str) -> anyhow::Result<()> {
    let manifest = load_manifest(base_name)?;
    let input = load_input(base_name)?;

    let profiles = Profiles::load(&project_path("models/profiles.toml"))?;
    let resolved = profiles.resolve(Some(&manifest.profile))?;
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

    Ok(())
}

#[test]
#[ignore = "requires local downloaded model assets; run `just test-fixtures`"]
fn fixture_regression_eu_pii_windowed() -> anyhow::Result<()> {
    run_fixture("eu_pii_windowed")
}

#[test]
#[ignore = "requires local downloaded model assets; run `just test-fixtures`"]
fn fixture_regression_xenova_ner_windowed() -> anyhow::Result<()> {
    run_fixture("xenova_ner_windowed")
}
