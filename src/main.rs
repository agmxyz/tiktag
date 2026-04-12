mod anonymize;
mod cli;
mod decode;
mod download;
#[cfg(test)]
mod fixture_tests;
mod model_bundle;
mod profiles;
mod runtime;
mod window;

use std::collections::BTreeMap;
use std::io::Read;

use anyhow::{Context, anyhow};
use log::info;
use serde::Serialize;

use crate::anonymize::{AnonymizationResult, Replacement};
use crate::profiles::ResolvedProfile;
use crate::runtime::{InferenceResult, InferenceTimings, LoadResult, LoadTimings, ModelRuntime};

/// Structured output for --json mode. Includes anonymized text, replacement metadata, and stats.
#[derive(Debug, Serialize)]
struct JsonOutput {
    profile: String,
    anonymized_text: String,
    replacements: Vec<Replacement>,
    placeholder_map: BTreeMap<String, String>,
    stats: JsonStats,
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
    let profiles = profiles::Profiles::load_internal()?;
    let resolved = profiles.resolve_default();

    info!(
        "selected model='{}' hf_repo='{}' model_dir='{}' max_tokens={} overlap_tokens={}",
        resolved.name,
        resolved.hf_repo,
        resolved.model_dir.display(),
        resolved.max_tokens,
        resolved.overlap_tokens,
    );

    let input_text = resolve_input_text(&args)?;

    let LoadResult {
        mut runtime,
        timings: load_timings,
    } = ModelRuntime::load(&resolved)?;
    let InferenceResult {
        entities,
        sequence_len,
        window_count,
        timings: infer_timings,
    } = runtime.infer(&input_text, args.show_tokens)?;
    let anonymized = crate::anonymize::anonymize(&input_text, &entities)?;

    if args.json {
        print_json(
            &resolved,
            sequence_len,
            window_count,
            load_timings,
            infer_timings,
            anonymized,
        )?;
    } else {
        print_text(&anonymized.anonymized_text);
    }

    Ok(())
}

fn run_download(_args: cli::DownloadArgs) -> anyhow::Result<()> {
    crate::download::download()
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
    resolved: &ResolvedProfile,
    sequence_len: usize,
    window_count: usize,
    load_timings: LoadTimings,
    infer_timings: InferenceTimings,
    anonymized: AnonymizationResult,
) -> anyhow::Result<()> {
    let payload = build_json_output(
        resolved,
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
    resolved: &ResolvedProfile,
    sequence_len: usize,
    window_count: usize,
    load_timings: LoadTimings,
    infer_timings: InferenceTimings,
    anonymized: AnonymizationResult,
) -> JsonOutput {
    JsonOutput {
        profile: resolved.name.clone(),
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

    use super::build_json_output;
    use crate::anonymize::{AnonymizationResult, PlaceholderFamily, Replacement};
    use crate::profiles::ResolvedProfile;
    use crate::runtime::{InferenceTimings, LoadTimings};

    fn resolved_profile() -> ResolvedProfile {
        ResolvedProfile {
            name: "eu_pii".to_owned(),
            hf_repo: "bardsai/eu-pii-anonimization-multilang".to_owned(),
            model_dir: PathBuf::from("models/eu-pii-anonimization-multilang"),
            max_tokens: 512,
            overlap_tokens: 128,
        }
    }

    fn sample_load_timings() -> LoadTimings {
        LoadTimings { total_ms: 46.0 }
    }

    fn sample_infer_timings(total_ms: f64) -> InferenceTimings {
        InferenceTimings { total_ms }
    }

    fn sample_anonymization_result() -> AnonymizationResult {
        AnonymizationResult {
            anonymized_text: "[PERSON_1] emailed [EMAIL_1]".to_owned(),
            replacements: vec![
                Replacement {
                    start: 0,
                    end: 5,
                    family: PlaceholderFamily::Person,
                    placeholder: "[PERSON_1]".to_owned(),
                    original: "Maria".to_owned(),
                },
                Replacement {
                    start: 14,
                    end: 31,
                    family: PlaceholderFamily::Email,
                    placeholder: "[EMAIL_1]".to_owned(),
                    original: "maria@example.com".to_owned(),
                },
            ],
            placeholder_map: BTreeMap::from([
                ("[PERSON_1]".to_owned(), "Maria".to_owned()),
                ("[EMAIL_1]".to_owned(), "maria@example.com".to_owned()),
            ]),
            counts_by_family: BTreeMap::from([("PERSON".to_owned(), 1), ("EMAIL".to_owned(), 1)]),
            detected_entity_count: 3,
            accepted_replacement_count: 2,
        }
    }

    #[test]
    fn json_payload_includes_anonymized_text_and_stats() {
        let payload = build_json_output(
            &resolved_profile(),
            27,
            1,
            sample_load_timings(),
            sample_infer_timings(11.5),
            sample_anonymization_result(),
        );

        let json = serde_json::to_value(payload).expect("json");
        assert_eq!(
            json["anonymized_text"],
            Value::from("[PERSON_1] emailed [EMAIL_1]")
        );
        assert_eq!(json["stats"]["accepted_replacement_count"], Value::from(2));
        assert_eq!(json["placeholder_map"]["[PERSON_1]"], Value::from("Maria"));
        assert_eq!(json["stats"]["timings"]["load"]["total_ms"], Value::from(46.0));
        assert_eq!(json["stats"]["timings"]["infer"]["total_ms"], Value::from(11.5));
    }

    #[test]
    fn json_payload_includes_counts_and_total_timings() {
        let payload = build_json_output(
            &resolved_profile(),
            976,
            3,
            sample_load_timings(),
            sample_infer_timings(16.5),
            sample_anonymization_result(),
        );

        let json = serde_json::to_value(payload).expect("json");
        assert_eq!(json["stats"]["window_count"], Value::from(3));
        assert_eq!(json["stats"]["counts_by_family"]["PERSON"], Value::from(1));
        assert_eq!(json["stats"]["timings"]["load"]["total_ms"], Value::from(46.0));
        assert_eq!(json["stats"]["timings"]["infer"]["total_ms"], Value::from(16.5));
        assert!(json["stats"].get("timings_ms").is_none());
    }
}
