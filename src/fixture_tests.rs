// End-to-end regression fixtures. Each fixture pairs an input markdown file
// with a TOML manifest listing expected placeholders, min window count, and
// forbidden literals. Tests are `#[ignore]` by default because they need the
// real ~500MB model bundle on disk; run `just test-fixtures` after a
// `just download`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{BUILTIN_PROFILE_NAME, Tiktag, missing_model_files};

const BUILTIN_MODEL_DIR: &str = "models/distilbert-base-multilingual-cased-ner-hrl";

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

#[derive(Debug, Deserialize)]
struct SyntheticFixture {
    name: String,
    text: String,
    expected: Vec<SyntheticLabel>,
    windows: usize,
}

#[derive(Debug, Deserialize)]
struct SyntheticLabel {
    start: usize,
    end: usize,
    family: String,
}

#[derive(Debug, Deserialize)]
struct BehaviorSnapshot {
    windows: usize,
    replacements: Vec<BehaviorReplacement>,
    placeholder_map_sha256: std::collections::BTreeMap<String, String>,
    anonymized_text_sha256: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct BehaviorReplacement {
    start: usize,
    end: usize,
    family: String,
    placeholder: String,
    original_sha256: String,
}

const SYNTHETIC_FIXTURES: [&str; 6] = [
    "short",
    "near_limit",
    "boundary",
    "multi_window",
    "unicode_repeated",
    "no_entity",
];

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
    let missing = missing_model_files(&model_dir)
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();

    if missing.is_empty() {
        return Ok(());
    }

    bail!(
        "fixture tests for profile '{BUILTIN_PROFILE_NAME}' require local model assets; missing: {}. Run `cargo run --locked --release -- download` first.",
        missing.join(", ")
    );
}

fn load_synthetic_fixture(name: &str) -> anyhow::Result<SyntheticFixture> {
    let path = project_path(&format!("performance/fixtures/{name}.json"));
    let contents = fs::read_to_string(&path)
        .with_context(|| format!("failed to read synthetic fixture {}", path.display()))?;
    let fixture: SyntheticFixture = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse synthetic fixture {}", path.display()))?;
    anyhow::ensure!(
        fixture.name == name,
        "fixture filename/name mismatch for {name}"
    );
    Ok(fixture)
}

fn sha256_hex(value: &[u8]) -> String {
    format!("{:x}", Sha256::digest(value))
}

fn semantic_replacements(output: &crate::TiktagOutput) -> Vec<BehaviorReplacement> {
    output
        .anonymization
        .replacements
        .iter()
        .map(|replacement| BehaviorReplacement {
            start: replacement.start,
            end: replacement.end,
            family: replacement.family.to_string(),
            placeholder: replacement.placeholder.clone(),
            original_sha256: sha256_hex(replacement.original.as_bytes()),
        })
        .collect()
}

fn confidence_values(output: &crate::TiktagOutput) -> Vec<(usize, usize, String, f32)> {
    output
        .anonymization
        .replacements
        .iter()
        .map(|replacement| {
            (
                replacement.start,
                replacement.end,
                replacement.family.to_string(),
                replacement.score,
            )
        })
        .collect()
}

fn placeholder_map_hashes(
    output: &crate::TiktagOutput,
) -> std::collections::BTreeMap<String, String> {
    output
        .anonymization
        .placeholder_map
        .iter()
        .map(|(placeholder, original)| (placeholder.clone(), sha256_hex(original.as_bytes())))
        .collect()
}

fn assert_semantic_behavior(
    fixture_name: &str,
    fixture: &SyntheticFixture,
    snapshot: &BehaviorSnapshot,
    output: &crate::TiktagOutput,
) {
    let text = fixture.text.as_str();
    let replacements = &output.anonymization.replacements;
    let mut identities = std::collections::BTreeMap::<(String, String), String>::new();
    let mut placeholders = std::collections::BTreeSet::<String>::new();

    for replacement in replacements {
        assert!(
            replacement.start < replacement.end,
            "empty replacement in {fixture_name}"
        );
        assert!(
            text.get(replacement.start..replacement.end) == Some(replacement.original.as_str()),
            "replacement offsets must be valid UTF-8 byte offsets in {fixture_name}"
        );
        assert!(
            output
                .anonymization
                .placeholder_map
                .get(&replacement.placeholder)
                == Some(&replacement.original),
            "placeholder map identity changed in {fixture_name}"
        );
        placeholders.insert(replacement.placeholder.clone());

        let identity = (
            replacement.family.to_string(),
            replacement.original.trim().to_owned(),
        );
        if let Some(previous) = identities.get(&identity) {
            assert_eq!(
                previous, &replacement.placeholder,
                "repeated value received a different placeholder in {fixture_name}"
            );
        } else {
            identities.insert(identity, replacement.placeholder.clone());
        }
    }

    assert!(
        replacements
            .windows(2)
            .all(|pair| pair[0].end <= pair[1].start),
        "replacements must be sorted and non-overlapping in {fixture_name}"
    );
    let unique_replacements = replacements
        .iter()
        .map(|r| (r.start, r.end, r.family.to_string(), r.placeholder.clone()))
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        unique_replacements.len(),
        replacements.len(),
        "duplicate replacement in {fixture_name}"
    );
    let map_placeholders = output
        .anonymization
        .placeholder_map
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        placeholders, map_placeholders,
        "placeholder map keys must match accepted replacements in {fixture_name}"
    );
    assert_eq!(
        semantic_replacements(output),
        snapshot.replacements,
        "span/family/placeholder behavior changed for {fixture_name}"
    );
    assert_eq!(
        placeholder_map_hashes(output),
        snapshot.placeholder_map_sha256,
        "placeholder-to-original identity changed for {fixture_name}"
    );
    assert_eq!(
        sha256_hex(output.anonymization.anonymized_text.as_bytes()),
        snapshot.anonymized_text_sha256,
        "masked text changed for {fixture_name}"
    );
    assert_eq!(
        output.window_count, snapshot.windows,
        "window count changed for {fixture_name}"
    );
    assert_eq!(
        output.window_count, fixture.windows,
        "window count differs from fixture contract for {fixture_name}"
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

/// Pinned semantic output is separate from the independent fixture labels.
/// Alternating long/short calls catches leaked tokenizer truncation settings.
#[test]
fn synthetic_fixture_labels_use_valid_utf8_byte_offsets() -> anyhow::Result<()> {
    for name in SYNTHETIC_FIXTURES {
        let fixture = load_synthetic_fixture(name)?;
        let mut previous_end = 0;
        for label in fixture.expected {
            anyhow::ensure!(label.start < label.end, "empty label in {name}");
            anyhow::ensure!(
                matches!(
                    label.family.as_str(),
                    "PERSON" | "ORG" | "LOCATION" | "EMAIL_ADDRESS"
                ),
                "unsupported label family '{}' in {name}",
                label.family
            );
            anyhow::ensure!(
                fixture.text.get(label.start..label.end).is_some(),
                "label [{},{}) is not a valid UTF-8 byte range in {name}",
                label.start,
                label.end
            );
            anyhow::ensure!(
                label.start >= previous_end,
                "overlapping or unsorted labels in {name}"
            );
            previous_end = label.end;
        }
    }
    Ok(())
}

/// Recorded draft behavior, deliberately separate from independent labels.
/// The ignored model test validates this snapshot against pinned local assets.
#[test]
#[ignore = "requires local downloaded model assets"]
fn fixture_regression_synthetic_pipeline() -> anyhow::Result<()> {
    require_local_model_assets()?;
    let observed: std::collections::BTreeMap<String, BehaviorSnapshot> =
        serde_json::from_str(include_str!("../performance/fixtures/behavior_spans.json"))?;
    let mut engine = Tiktag::new(&project_path("models/profiles.toml"))?;
    let short = load_synthetic_fixture("short")?;
    let short_snapshot = observed
        .get("short")
        .context("missing short behavior snapshot")?;
    let initial_short = engine.anonymize(&short.text)?;
    assert_semantic_behavior("short", &short, short_snapshot, &initial_short);

    for name in [
        "multi_window",
        "boundary",
        "near_limit",
        "unicode_repeated",
        "no_entity",
    ] {
        let fixture = load_synthetic_fixture(name)?;
        let snapshot = observed
            .get(name)
            .with_context(|| format!("missing {name} behavior snapshot"))?;
        assert_eq!(
            fixture.windows, snapshot.windows,
            "snapshot window contract changed for {name}"
        );
        let output = engine.anonymize(&fixture.text)?;
        assert_semantic_behavior(name, &fixture, snapshot, &output);
        #[cfg(feature = "profiling")]
        {
            let (measured, _) = engine.anonymize_measured(&fixture.text)?;
            assert_eq!(
                semantic_replacements(&measured),
                semantic_replacements(&output),
                "instrumentation changed semantic output for {name}"
            );
            assert_eq!(
                confidence_values(&measured),
                confidence_values(&output),
                "instrumentation changed confidence for {name}"
            );
        }
        let subsequent_short = engine.anonymize(&short.text)?;
        assert_semantic_behavior(
            "short after long call",
            &short,
            short_snapshot,
            &subsequent_short,
        );
        assert_eq!(
            confidence_values(&subsequent_short),
            confidence_values(&initial_short),
            "confidence changed for short call after {name}"
        );
    }
    Ok(())
}
