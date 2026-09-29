//! One fresh process/session per fixture. Driver: scripts/benchmark.py.
use anyhow::{Result, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, process::Command, time::Instant};
use tiktag::{Profiles, Tiktag, TiktagOutput};
use tokenizers::{Tokenizer, utils::truncation::TruncationParams};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models/profiles.toml")]
    profiles: PathBuf,
    #[arg(long)]
    fixture: PathBuf,
    #[arg(long, default_value_t = 30)]
    samples: usize,
    #[arg(long, default_value_t = 5)]
    warmup: usize,
    #[arg(long)]
    trace: Option<PathBuf>,
    /// Stage instrumentation is separate from the uninstrumented latency sample loop.
    #[arg(long, default_value_t = 5)]
    stage_samples: usize,
}

#[derive(Deserialize)]
struct Fixture {
    name: String,
    text: String,
    expected: Vec<Span>,
    min_tokens: usize,
    max_tokens: usize,
    windows: usize,
    #[serde(default)]
    crosses_first_window: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
    family: String,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct SemanticReplacement {
    start: usize,
    end: usize,
    family: String,
    placeholder: String,
    original_sha256: String,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct SemanticSnapshot {
    replacements: Vec<SemanticReplacement>,
    placeholder_map_sha256: std::collections::BTreeMap<String, String>,
    anonymized_text_sha256: String,
    sequence_len: usize,
    windows: usize,
}

#[derive(Debug, Serialize, PartialEq)]
struct ConfidenceScore {
    start: usize,
    end: usize,
    family: String,
    placeholder: String,
    score: f32,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn semantic_snapshot(output: &TiktagOutput) -> SemanticSnapshot {
    let replacements = output
        .anonymization
        .replacements
        .iter()
        .map(|replacement| SemanticReplacement {
            start: replacement.start,
            end: replacement.end,
            family: replacement.family.to_string(),
            placeholder: replacement.placeholder.clone(),
            original_sha256: digest(replacement.original.as_bytes()),
        })
        .collect();
    let placeholder_map_sha256 = output
        .anonymization
        .placeholder_map
        .iter()
        .map(|(placeholder, original)| (placeholder.clone(), digest(original.as_bytes())))
        .collect();

    SemanticSnapshot {
        replacements,
        placeholder_map_sha256,
        anonymized_text_sha256: digest(output.anonymization.anonymized_text.as_bytes()),
        sequence_len: output.sequence_len,
        windows: output.window_count,
    }
}

fn confidence_scores(output: &TiktagOutput) -> Vec<ConfidenceScore> {
    output
        .anonymization
        .replacements
        .iter()
        .map(|replacement| ConfidenceScore {
            start: replacement.start,
            end: replacement.end,
            family: replacement.family.to_string(),
            placeholder: replacement.placeholder.clone(),
            score: replacement.score,
        })
        .collect()
}

fn confidence_drift(reference: &[ConfidenceScore], current: &[ConfidenceScore]) -> (usize, f32) {
    let mut changed = 0;
    let mut max_delta = 0.0_f32;
    for (before, after) in reference.iter().zip(current) {
        let delta = (after.score - before.score).abs();
        if delta != 0.0 {
            changed += 1;
            max_delta = max_delta.max(delta);
        }
    }
    (changed, max_delta)
}

fn record_confidence_drift(
    reference: &[ConfidenceScore],
    output: &TiktagOutput,
    comparisons_with_drift: &mut usize,
    max_confidence_delta: &mut f32,
) {
    let (changed, max_delta) = confidence_drift(reference, &confidence_scores(output));
    if changed > 0 {
        *comparisons_with_drift += 1;
    }
    *max_confidence_delta = max_confidence_delta.max(max_delta);
}

fn rss_bytes() -> Option<u64> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|kb| kb * 1024)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.samples > 0 && args.warmup > 0 && args.stage_samples > 0,
        "counts must be positive"
    );
    // No logger is installed, and runtime logs contain no fixture or entity values.
    let fixture_bytes = fs::read(&args.fixture)?;
    let fixture: Fixture = serde_json::from_slice(&fixture_bytes)?;
    let rss_before = rss_bytes();
    let (mut engine, initialization) = Tiktag::new_measured(&args.profiles, args.trace.as_deref())?;
    let rss_initialized = rss_bytes();
    let first_start = Instant::now();
    let first = engine.anonymize(&fixture.text)?;
    let first_ms = first_start.elapsed().as_secs_f64() * 1000.0;
    let rss_first = rss_bytes();
    ensure!(
        (fixture.min_tokens..=fixture.max_tokens).contains(&first.sequence_len),
        "fixture token count changed: {}",
        first.sequence_len
    );
    ensure!(
        first.window_count == fixture.windows,
        "fixture window count changed: {}",
        first.window_count
    );
    let semantic = semantic_snapshot(&first);
    let confidence = confidence_scores(&first);
    let mut confidence_comparisons_with_drift = 0usize;
    let mut max_confidence_delta = 0.0_f32;
    for _ in 0..args.warmup {
        let output = engine.anonymize(&fixture.text)?;
        ensure!(
            semantic_snapshot(&output) == semantic,
            "warmup semantics changed"
        );
        record_confidence_drift(
            &confidence,
            &output,
            &mut confidence_comparisons_with_drift,
            &mut max_confidence_delta,
        );
    }
    let mut warm_ms = Vec::with_capacity(args.samples);
    for _ in 0..args.samples {
        let start = Instant::now();
        let output = engine.anonymize(&fixture.text)?;
        warm_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        ensure!(
            semantic_snapshot(&output) == semantic,
            "sample semantics changed"
        );
        record_confidence_drift(
            &confidence,
            &output,
            &mut confidence_comparisons_with_drift,
            &mut max_confidence_delta,
        );
    }
    let mut stages = Vec::new();
    for _ in 0..args.stage_samples {
        let (output, timing) = engine.anonymize_measured(&fixture.text)?;
        ensure!(
            semantic_snapshot(&output) == semantic,
            "instrumentation changed semantics"
        );
        record_confidence_drift(
            &confidence,
            &output,
            &mut confidence_comparisons_with_drift,
            &mut max_confidence_delta,
        );
        stages.push(timing);
    }
    let trace_path = if args.trace.is_some() {
        Some(engine.end_profiling()?)
    } else {
        None
    };
    let rss_warm = rss_bytes();
    let actual: Vec<Span> = first
        .anonymization
        .replacements
        .iter()
        .map(|r| Span {
            start: r.start,
            end: r.end,
            family: r.family.to_string(),
        })
        .collect();
    for span in fixture.expected.iter().chain(actual.iter()) {
        ensure!(
            span.start < span.end && fixture.text.get(span.start..span.end).is_some(),
            "invalid UTF-8 byte span"
        );
    }
    ensure!(
        actual.windows(2).all(|w| w[0].end <= w[1].start),
        "duplicate or overlapping replacements"
    );
    let missed_labels: Vec<_> = fixture
        .expected
        .iter()
        .filter(|s| !actual.contains(s))
        .collect();
    let incorrect_outputs: Vec<_> = actual
        .iter()
        .filter(|s| !fixture.expected.contains(s))
        .collect();
    let wrong_family_actuals: Vec<_> = actual
        .iter()
        .filter(|observed| {
            fixture.expected.iter().any(|expected| {
                expected.start == observed.start
                    && expected.end == observed.end
                    && expected.family != observed.family
            })
        })
        .cloned()
        .collect();
    let wrong_family_masks: Vec<_> = wrong_family_actuals
        .iter()
        .filter_map(|observed| {
            fixture
                .expected
                .iter()
                .find(|expected| {
                    expected.start == observed.start
                        && expected.end == observed.end
                        && expected.family != observed.family
                })
                .map(|expected| json!({"expected": expected, "actual": observed}))
        })
        .collect();
    let partial_intervals: Vec<_> = actual
        .iter()
        .filter(|observed| {
            fixture.expected.iter().any(|expected| {
                observed.start < expected.end
                    && expected.start < observed.end
                    && (observed.start != expected.start || observed.end != expected.end)
            }) && !wrong_family_actuals.contains(observed)
        })
        .collect();
    let spurious_outputs: Vec<_> = actual
        .iter()
        .filter(|observed| {
            !fixture
                .expected
                .iter()
                .any(|expected| observed.start < expected.end && expected.start < observed.end)
        })
        .collect();
    // Inspection below is outside timing and follows RSS snapshots. Driver reports
    // whole-process peak separately; this extra tokenizer may contribute to that peak.
    let profile = Profiles::load(&args.profiles)?.resolve_default();
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(profile.model_dir.join("config.json"))?)?;
    ensure!(
        profile.max_tokens as u64 <= config["max_position_embeddings"].as_u64().unwrap_or(0),
        "profile exceeds model context"
    );
    let mut tokenizer = Tokenizer::from_file(profile.model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: profile.max_tokens,
            stride: profile.overlap_tokens,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let encoding = tokenizer
        .encode(fixture.text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let first_end = encoding
        .get_offsets()
        .iter()
        .map(|(_, end)| *end)
        .max()
        .unwrap_or(0);
    ensure!(
        !fixture.crosses_first_window
            || fixture
                .expected
                .iter()
                .any(|s| s.start < first_end && s.end > first_end),
        "expected entity does not cross first window boundary"
    );
    let processed_tokens = encoding.len()
        + encoding
            .get_overflowing()
            .iter()
            .map(|e| e.len())
            .sum::<usize>();
    let mut sorted = warm_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let p95 = sorted[(sorted.len() * 95).div_ceil(100) - 1];
    let total_seconds = warm_ms.iter().sum::<f64>() / 1000.0;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "fixture": fixture.name, "fixture_sha256": digest(&fixture_bytes),
            "input_bytes": fixture.text.len(), "input_tokens_including_special": first.sequence_len,
            "processed_tokens_including_overlap_and_special": processed_tokens, "windows": first.window_count,
            "initialization": initialization, "first_call_ms": first_ms,
            "warmup_count": args.warmup, "sample_count": args.samples,
            "warm_ms": warm_ms, "median_ms": median, "p95_ms": p95,
            "throughput": {"documents_per_second": args.samples as f64 / total_seconds,
                "input_tokens_per_second": args.samples as f64 * first.sequence_len as f64 / total_seconds,
                "input_bytes_per_second": args.samples as f64 * fixture.text.len() as f64 / total_seconds},
            "stage_samples": stages, "trace_path": trace_path, "ort_build_info": ort::info(),
            "rss_snapshots_bytes": {"before": rss_before, "initialized": rss_initialized, "first_call": rss_first, "warm": rss_warm},
            "semantic": semantic,
            "confidence": confidence,
            "confidence_drift_within_run": {
                "comparisons_with_drift": confidence_comparisons_with_drift,
                "max_absolute_score_delta": max_confidence_delta
            },
            "quality": {"expected": fixture.expected, "actual": actual,
                "missed_labels": missed_labels, "incorrect_outputs": incorrect_outputs,
                "wrong_family_masks": wrong_family_masks, "partial_intervals": partial_intervals,
                "spurious_outputs": spurious_outputs,
                "exact_matches": fixture.expected.len() - missed_labels.len()},
            "window_config": {"max_tokens": profile.max_tokens, "overlap_tokens": profile.overlap_tokens, "first_window_end_byte": first_end}
        }))?
    );
    Ok(())
}
