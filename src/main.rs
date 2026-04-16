mod cli;
mod download;

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, anyhow};
use log::info;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tiktag::{AnonymizationResult, REQUIRED_MODEL_FILES, Replacement, Tiktag};

const JSON_SCHEMA_VERSION: u32 = 1;
const INTERNAL_PROFILES_PATH: &str = "models/profiles.toml";

/// Structured output for --json mode. Safe by default: no reversible replacement metadata.
#[derive(Debug, Serialize)]
struct JsonOutput {
    schema_version: u32,
    provenance: JsonProvenance,
    profile: String,
    anonymized_text: String,
    stats: JsonStats,
}

/// Structured output for --debug-json mode. Includes reversible replacement metadata.
#[derive(Debug, Serialize)]
struct DebugJsonOutput {
    schema_version: u32,
    provenance: JsonProvenance,
    profile: String,
    anonymized_text: String,
    replacements: Vec<Replacement>,
    placeholder_map: BTreeMap<String, String>,
    stats: JsonStats,
}

#[derive(Debug, Clone)]
struct CliProfileMetadata {
    profile: String,
    hf_repo: String,
    model_dir: PathBuf,
}

#[derive(Debug, Serialize)]
struct JsonProvenance {
    app_version: String,
    hf_repo: String,
    bundle_sha256: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct LoadTimings {
    total_ms: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct InferenceTimings {
    total_ms: f64,
}

#[derive(Debug, Serialize)]
struct JsonTimings {
    load: LoadTimings,
    infer: InferenceTimings,
}

#[derive(Debug, Serialize)]
struct JsonStats {
    sequence_len: usize,
    window_count: usize,
    detected_entity_count: usize,
    accepted_replacement_count: usize,
    counts_by_family: BTreeMap<String, usize>,
    timings: JsonTimings,
}

fn main() -> anyhow::Result<()> {
    let command = cli::parse();
    let show_tokens = matches!(&command, cli::Command::Run(args) if args.show_tokens);
    init_logging(show_tokens);

    match command {
        cli::Command::Run(args) => run_inference(args),
        cli::Command::Download(args) => run_download(args),
    }
}

fn run_inference(args: cli::RunArgs) -> anyhow::Result<()> {
    let input_text = resolve_input_text(&args)?;

    let load_start = Instant::now();
    let mut tiktag = Tiktag::new(Path::new(INTERNAL_PROFILES_PATH))?;
    let load_timings = LoadTimings {
        total_ms: finish_timing("load.total", load_start),
    };
    let profile = CliProfileMetadata {
        profile: tiktag.profile_name().to_owned(),
        hf_repo: tiktag.hf_repo().to_owned(),
        model_dir: tiktag.model_dir().to_path_buf(),
    };

    info!(
        "selected model='{}' hf_repo='{}' model_dir='{}'",
        profile.profile,
        profile.hf_repo,
        profile.model_dir.display(),
    );

    let infer_start = Instant::now();
    let output = tiktag.anonymize(&input_text)?;
    let infer_timings = InferenceTimings {
        total_ms: finish_timing("infer.total", infer_start),
    };

    if args.debug_json {
        print_debug_json(
            &profile,
            output.sequence_len,
            output.window_count,
            load_timings,
            infer_timings,
            output.anonymization,
        )?;
    } else if args.json {
        print_json(
            &profile,
            output.sequence_len,
            output.window_count,
            load_timings,
            infer_timings,
            output.anonymization,
        )?;
    } else {
        print_text(&output.anonymization.anonymized_text);
    }

    Ok(())
}

fn run_download(_args: cli::DownloadArgs) -> anyhow::Result<()> {
    download::download()
}

fn resolve_input_text(args: &cli::RunArgs) -> anyhow::Result<String> {
    if args.stdin {
        let mut text = String::new();
        std::io::stdin()
            .read_to_string(&mut text)
            .context("failed to read input from stdin")?;
        return Ok(text);
    }

    args.text
        .clone()
        .ok_or_else(|| anyhow!("missing input text; pass <TEXT> or use --stdin"))
}

fn print_text(anonymized_text: &str) {
    println!("{anonymized_text}");
}

fn print_json(
    profile: &CliProfileMetadata,
    sequence_len: usize,
    window_count: usize,
    load_timings: LoadTimings,
    infer_timings: InferenceTimings,
    anonymized: AnonymizationResult,
) -> anyhow::Result<()> {
    let provenance = build_json_provenance(profile)?;
    let payload = build_json_output(
        provenance,
        profile,
        sequence_len,
        window_count,
        load_timings,
        infer_timings,
        anonymized,
    );

    let json = serde_json::to_string_pretty(&payload)?;
    println!("{json}");
    Ok(())
}

fn print_debug_json(
    profile: &CliProfileMetadata,
    sequence_len: usize,
    window_count: usize,
    load_timings: LoadTimings,
    infer_timings: InferenceTimings,
    anonymized: AnonymizationResult,
) -> anyhow::Result<()> {
    let provenance = build_json_provenance(profile)?;
    let payload = build_debug_json_output(
        provenance,
        profile,
        sequence_len,
        window_count,
        load_timings,
        infer_timings,
        anonymized,
    );

    let json = serde_json::to_string_pretty(&payload)?;
    println!("{json}");
    Ok(())
}

fn build_json_output(
    provenance: JsonProvenance,
    profile: &CliProfileMetadata,
    sequence_len: usize,
    window_count: usize,
    load_timings: LoadTimings,
    infer_timings: InferenceTimings,
    anonymized: AnonymizationResult,
) -> JsonOutput {
    JsonOutput {
        schema_version: JSON_SCHEMA_VERSION,
        provenance,
        profile: profile.profile.clone(),
        anonymized_text: anonymized.anonymized_text,
        stats: JsonStats {
            sequence_len,
            window_count,
            detected_entity_count: anonymized.detected_entity_count,
            accepted_replacement_count: anonymized.accepted_replacement_count,
            counts_by_family: anonymized.counts_by_family,
            timings: JsonTimings {
                load: load_timings,
                infer: infer_timings,
            },
        },
    }
}

fn build_debug_json_output(
    provenance: JsonProvenance,
    profile: &CliProfileMetadata,
    sequence_len: usize,
    window_count: usize,
    load_timings: LoadTimings,
    infer_timings: InferenceTimings,
    anonymized: AnonymizationResult,
) -> DebugJsonOutput {
    DebugJsonOutput {
        schema_version: JSON_SCHEMA_VERSION,
        provenance,
        profile: profile.profile.clone(),
        anonymized_text: anonymized.anonymized_text,
        replacements: anonymized.replacements,
        placeholder_map: anonymized.placeholder_map,
        stats: JsonStats {
            sequence_len,
            window_count,
            detected_entity_count: anonymized.detected_entity_count,
            accepted_replacement_count: anonymized.accepted_replacement_count,
            counts_by_family: anonymized.counts_by_family,
            timings: JsonTimings {
                load: load_timings,
                infer: infer_timings,
            },
        },
    }
}

fn build_json_provenance(profile: &CliProfileMetadata) -> anyhow::Result<JsonProvenance> {
    let mut hasher = Sha256::new();
    hash_bundle_file(
        &mut hasher,
        INTERNAL_PROFILES_PATH,
        Path::new(INTERNAL_PROFILES_PATH),
    )?;

    for relative_path in REQUIRED_MODEL_FILES {
        hash_bundle_file(
            &mut hasher,
            relative_path,
            &profile.model_dir.join(relative_path),
        )?;
    }

    Ok(JsonProvenance {
        app_version: env!("CARGO_PKG_VERSION").to_owned(),
        hf_repo: profile.hf_repo.clone(),
        bundle_sha256: format!("{:x}", hasher.finalize()),
    })
}

fn hash_bundle_file(hasher: &mut Sha256, label: &str, path: &Path) -> anyhow::Result<()> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    hasher.update(label.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    Ok(())
}

fn finish_timing(stage: &str, started_at: Instant) -> f64 {
    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1_000.0;
    info!(target: "timing", "{stage}: {elapsed_ms:.2} ms");
    elapsed_ms
}

fn init_logging(show_tokens: bool) {
    let mut builder =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"));
    if show_tokens {
        // Ensure token traces are visible when --show-tokens is requested.
        builder.filter_module("tokens", log::LevelFilter::Debug);
    }
    builder.format_timestamp_millis();
    let _ = builder.try_init();
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use serde_json::Value;

    use super::{CliProfileMetadata, build_debug_json_output, build_json_output};
    use crate::{InferenceTimings, JsonProvenance, LoadTimings};
    use tiktag::{AnonymizationResult, PlaceholderFamily, Replacement};

    fn profile_metadata() -> CliProfileMetadata {
        CliProfileMetadata {
            profile: "distilbert_ner_hrl".to_owned(),
            hf_repo: "Xenova/distilbert-base-multilingual-cased-ner-hrl".to_owned(),
            model_dir: PathBuf::from("models/distilbert-base-multilingual-cased-ner-hrl"),
        }
    }

    fn sample_load_timings() -> LoadTimings {
        LoadTimings { total_ms: 46.0 }
    }

    fn sample_infer_timings(total_ms: f64) -> InferenceTimings {
        InferenceTimings { total_ms }
    }

    fn sample_provenance() -> JsonProvenance {
        JsonProvenance {
            app_version: "0.1.0".to_owned(),
            hf_repo: "Xenova/distilbert-base-multilingual-cased-ner-hrl".to_owned(),
            bundle_sha256: "abc123".to_owned(),
        }
    }

    fn sample_anonymization_result() -> AnonymizationResult {
        AnonymizationResult {
            anonymized_text: "[PERSON_1] works at [ORG_1]".to_owned(),
            replacements: vec![
                Replacement {
                    start: 0,
                    end: 5,
                    family: PlaceholderFamily::Person,
                    placeholder: "[PERSON_1]".to_owned(),
                    original: "Maria".to_owned(),
                    score: 0.95,
                },
                Replacement {
                    start: 15,
                    end: 21,
                    family: PlaceholderFamily::Org,
                    placeholder: "[ORG_1]".to_owned(),
                    original: "OpenAI".to_owned(),
                    score: 0.88,
                },
            ],
            placeholder_map: BTreeMap::from([
                ("[PERSON_1]".to_owned(), "Maria".to_owned()),
                ("[ORG_1]".to_owned(), "OpenAI".to_owned()),
            ]),
            counts_by_family: BTreeMap::from([("PERSON".to_owned(), 1), ("ORG".to_owned(), 1)]),
            detected_entity_count: 3,
            accepted_replacement_count: 2,
        }
    }

    #[test]
    fn json_payload_is_safe_by_default() {
        let payload = build_json_output(
            sample_provenance(),
            &profile_metadata(),
            27,
            1,
            sample_load_timings(),
            sample_infer_timings(11.5),
            sample_anonymization_result(),
        );

        let json = serde_json::to_value(payload).expect("json");
        assert_eq!(json["schema_version"], Value::from(1));
        assert_eq!(
            json["anonymized_text"],
            Value::from("[PERSON_1] works at [ORG_1]")
        );
        assert_eq!(json["stats"]["accepted_replacement_count"], Value::from(2));
        assert_eq!(json["provenance"]["app_version"], Value::from("0.1.0"));
        assert_eq!(
            json["stats"]["timings"]["load"]["total_ms"],
            Value::from(46.0)
        );
        assert_eq!(
            json["stats"]["timings"]["infer"]["total_ms"],
            Value::from(11.5)
        );
        assert!(json.get("placeholder_map").is_none());
        assert!(json.get("replacements").is_none());
    }

    #[test]
    fn json_payload_includes_counts_and_total_timings() {
        let payload = build_json_output(
            sample_provenance(),
            &profile_metadata(),
            976,
            3,
            sample_load_timings(),
            sample_infer_timings(16.5),
            sample_anonymization_result(),
        );

        let json = serde_json::to_value(payload).expect("json");
        assert_eq!(json["stats"]["window_count"], Value::from(3));
        assert_eq!(json["stats"]["counts_by_family"]["PERSON"], Value::from(1));
        assert_eq!(
            json["stats"]["timings"]["load"]["total_ms"],
            Value::from(46.0)
        );
        assert_eq!(
            json["stats"]["timings"]["infer"]["total_ms"],
            Value::from(16.5)
        );
        assert!(json["stats"].get("timings_ms").is_none());
    }

    #[test]
    fn debug_json_payload_includes_reversible_metadata() {
        let payload = build_debug_json_output(
            sample_provenance(),
            &profile_metadata(),
            27,
            1,
            sample_load_timings(),
            sample_infer_timings(11.5),
            sample_anonymization_result(),
        );

        let json = serde_json::to_value(payload).expect("json");
        assert_eq!(json["schema_version"], Value::from(1));
        assert_eq!(json["placeholder_map"]["[PERSON_1]"], Value::from("Maria"));
        assert_eq!(json["replacements"][0]["original"], Value::from("Maria"));
        assert_eq!(json["replacements"][0]["score"], Value::from(0.95_f32));
    }
}
