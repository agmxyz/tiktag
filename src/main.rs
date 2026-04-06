mod cli;
mod decode;
mod profiles;
mod runtime;

use std::collections::BTreeMap;

use anyhow::Context;
use clap::Parser;
use log::info;
use serde::Serialize;

use crate::decode::EntitySpan;
use crate::profiles::ResolvedProfile;
use crate::runtime::{InferenceResult, LoadResult, ModelRuntime};

/// Structured output for --json mode. Includes profile metadata, timings, and entities.
#[derive(Debug, Serialize)]
struct JsonOutput {
    profile: String,
    hf_repo: String,
    model_dir: String,
    max_tokens: usize,
    decode_strategy: crate::decode::DecodeStrategy,
    sequence_len: usize,
    timings_ms: BTreeMap<String, f64>,
    entities: Vec<EntitySpan>,
}

fn main() -> anyhow::Result<()> {
    init_logging();

    let args = cli::Args::parse();
    let profiles = profiles::Profiles::load(&args.profiles).with_context(|| {
        format!(
            "failed to load profiles from {}",
            args.profiles.as_path().display()
        )
    })?;
    let resolved = profiles.resolve(args.profile.as_deref())?;

    info!(
        "selected profile='{}' hf_repo='{}' model_dir='{}' max_tokens={} decode_strategy={}",
        resolved.name,
        resolved.hf_repo,
        resolved.model_dir.display(),
        resolved.max_tokens,
        resolved.decode_strategy
    );

    let LoadResult {
        mut runtime,
        timings_ms: load_timings_ms,
    } = ModelRuntime::load(&resolved)?;
    let InferenceResult {
        entities,
        sequence_len,
        timings_ms: infer_timings_ms,
    } = runtime.infer(&args.text, args.show_tokens)?;

    let mut timings_ms = load_timings_ms;
    timings_ms.extend(infer_timings_ms);

    if args.json {
        print_json(&resolved, sequence_len, timings_ms, entities)?;
    } else {
        print_human(&entities);
    }

    Ok(())
}

fn print_human(entities: &[EntitySpan]) {
    if entities.is_empty() {
        println!("No entities detected.");
        return;
    }

    for entity in entities {
        println!(
            "{} [{}..{}]: {}",
            entity.label, entity.start, entity.end, entity.text
        );
    }
}

fn print_json(
    resolved: &ResolvedProfile,
    sequence_len: usize,
    timings_ms: BTreeMap<String, f64>,
    entities: Vec<EntitySpan>,
) -> anyhow::Result<()> {
    let payload = JsonOutput {
        profile: resolved.name.clone(),
        hf_repo: resolved.hf_repo.clone(),
        model_dir: resolved.model_dir.display().to_string(),
        max_tokens: resolved.max_tokens,
        decode_strategy: resolved.decode_strategy,
        sequence_len,
        timings_ms,
        entities,
    };

    let json = serde_json::to_string_pretty(&payload)?;
    println!("{json}");
    Ok(())
}

fn init_logging() {
    let mut builder =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"));
    builder.format_timestamp_millis();
    let _ = builder.try_init();
}
