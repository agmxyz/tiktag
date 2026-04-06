// ONNX model loading and inference. This is the core runtime:
//   1. Validate the model bundle (tokenizer.json, config.json, onnx/model_quantized.onnx)
//   2. Load tokenizer, labels from config.json id2label, and ONNX session
//   3. Tokenize input text, run the ONNX graph, decode entity spans from logits
// All timings are collected into a BTreeMap for structured output.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, anyhow, bail};
use log::{debug, info};
use ndarray::{Array2, Ix3};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::TensorRef;
use serde_json::Value;
use tokenizers::Tokenizer;
use tokenizers::utils::truncation::{TruncationDirection, TruncationParams, TruncationStrategy};

use crate::decode::{DecodeStrategy, EntitySpan, argmax_label_indices, decode_entities};
use crate::profiles::ResolvedProfile;
use crate::window::WindowEntities;

/// Every model bundle must have these three files present.
const REQUIRED_MODEL_FILES: &[&str] =
    &["tokenizer.json", "config.json", "onnx/model_quantized.onnx"];

#[derive(Debug)]
pub struct LoadResult {
    pub runtime: ModelRuntime,
    pub timings_ms: BTreeMap<String, f64>,
}

#[derive(Debug)]
pub struct InferenceResult {
    pub entities: Vec<EntitySpan>,
    pub sequence_len: usize,
    pub window_count: usize,
    pub timings_ms: BTreeMap<String, f64>,
}

/// Holds everything needed to run inference: the tokenizer, label map, ONNX session,
/// and profile settings. Created via ModelRuntime::load().
#[derive(Debug)]
pub struct ModelRuntime {
    tokenizer: Tokenizer,
    labels: Vec<String>,
    session: Session,
    profile_name: String,
    max_tokens: usize,
    overlap_tokens: usize,
    decode_strategy: DecodeStrategy,
    /// BERT-family models need token_type_ids; DistilBERT doesn't. Detected at load time.
    has_token_type_ids: bool,
}

impl ModelRuntime {
    pub fn load(profile: &ResolvedProfile) -> anyhow::Result<LoadResult> {
        let mut timings_ms = BTreeMap::new();
        let total_start = Instant::now();

        validate_model_bundle(&profile.model_dir)?;

        let tokenizer_path = profile.model_dir.join("tokenizer.json");
        let config_path = profile.model_dir.join("config.json");
        let model_path = profile.model_dir.join("onnx/model_quantized.onnx");

        info!(
            "loading model assets for profile='{}' from '{}'",
            profile.name,
            profile.model_dir.display()
        );

        let tokenizer_start = Instant::now();
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
            anyhow!(
                "failed to load tokenizer at {}: {err}",
                tokenizer_path.display()
            )
        })?;
        // Disable the tokenizer's built-in truncation so we can enforce max_tokens ourselves
        // with a clear error message instead of silently dropping tokens.
        tokenizer.with_truncation(None).map_err(|err| {
            anyhow!(
                "failed to disable truncation in tokenizer {}: {err}",
                tokenizer_path.display()
            )
        })?;
        record_timing(&mut timings_ms, "load.tokenizer", tokenizer_start);

        let labels_start = Instant::now();
        let labels = load_labels(&config_path)?;
        record_timing(&mut timings_ms, "load.labels", labels_start);

        let ort_init_start = Instant::now();
        ort::init().commit();
        record_timing(&mut timings_ms, "load.ort_init", ort_init_start);

        let session_start = Instant::now();
        let session = Session::builder()
            .map_err(ort_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_error)?
            .commit_from_file(model_path)
            .map_err(ort_error)
            .context("failed to create ONNX Runtime session")?;
        record_timing(&mut timings_ms, "load.session", session_start);

        // Some ONNX exports (e.g. BERT-family) require a token_type_ids input; others
        // (e.g. DistilBERT) don't. Probe the session graph to decide at load time.
        let has_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        record_timing(&mut timings_ms, "load.total", total_start);

        Ok(LoadResult {
            runtime: Self {
                tokenizer,
                labels,
                session,
                profile_name: profile.name.clone(),
                max_tokens: profile.max_tokens,
                overlap_tokens: profile.overlap_tokens,
                decode_strategy: profile.decode_strategy,
                has_token_type_ids,
            },
            timings_ms,
        })
    }

    pub fn infer(&mut self, text: &str, show_tokens: bool) -> anyhow::Result<InferenceResult> {
        let mut timings_ms = BTreeMap::new();
        let total_start = Instant::now();
        info!("running inference on {} input bytes", text.len());

        // Tokenize without truncation to measure true sequence length.
        let tokenize_start = Instant::now();
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|err| anyhow!("tokenization failed: {err}"))?;
        let seq_len = encoding.len();
        if seq_len == 0 {
            bail!("tokenizer returned an empty encoding");
        }
        record_timing(&mut timings_ms, "infer.tokenize", tokenize_start);
        info!("encoded sequence length: {seq_len}");

        if seq_len <= self.max_tokens {
            // Fast path: single window, identical to pre-windowing behavior.
            let entities =
                self.infer_single_encoding(text, &encoding, show_tokens, &mut timings_ms)?;
            record_timing(&mut timings_ms, "infer.total", total_start);
            Ok(InferenceResult {
                entities,
                sequence_len: seq_len,
                window_count: 1,
                timings_ms,
            })
        } else if self.overlap_tokens == 0 {
            // Windowing disabled — preserve fail-fast behavior.
            ensure_sequence_within_limit(&self.profile_name, seq_len, self.max_tokens)?;
            unreachable!()
        } else {
            // Sliding-window inference.
            info!(
                "input exceeds max_tokens={}, using sliding window (overlap_tokens={})",
                self.max_tokens, self.overlap_tokens
            );
            let (entities, window_count) =
                self.infer_windowed(text, show_tokens, &mut timings_ms)?;
            record_timing(&mut timings_ms, "infer.total", total_start);
            Ok(InferenceResult {
                entities,
                sequence_len: seq_len,
                window_count,
                timings_ms,
            })
        }
    }

    /// Run ONNX inference on a single encoding and decode entities.
    /// Shared by both the single-pass fast path and each window of the windowed path.
    fn infer_single_encoding(
        &mut self,
        text: &str,
        encoding: &tokenizers::Encoding,
        show_tokens: bool,
        timings_ms: &mut BTreeMap<String, f64>,
    ) -> anyhow::Result<Vec<EntitySpan>> {
        let seq_len = encoding.len();

        let prepare_start = Instant::now();
        let input_ids = Array2::from_shape_vec(
            (1, seq_len),
            encoding.get_ids().iter().map(|&id| i64::from(id)).collect(),
        )?;
        let attention_mask = Array2::from_shape_vec(
            (1, seq_len),
            encoding
                .get_attention_mask()
                .iter()
                .map(|&mask| i64::from(mask))
                .collect(),
        )?;
        record_timing(timings_ms, "infer.prepare", prepare_start);

        let run_start = Instant::now();
        let outputs = if self.has_token_type_ids {
            let token_type_ids = Array2::<i64>::zeros((1, seq_len));
            self.session.run(ort::inputs! {
                "input_ids" => TensorRef::from_array_view(input_ids.view())?,
                "attention_mask" => TensorRef::from_array_view(attention_mask.view())?,
                "token_type_ids" => TensorRef::from_array_view(token_type_ids.view())?,
            })?
        } else {
            self.session.run(ort::inputs! {
                "input_ids" => TensorRef::from_array_view(input_ids.view())?,
                "attention_mask" => TensorRef::from_array_view(attention_mask.view())?,
            })?
        };
        record_timing(timings_ms, "infer.onnx", run_start);

        let logits = outputs[0]
            .try_extract_array::<f32>()
            .context("failed to extract logits as f32 tensor")?;
        let logits = logits
            .into_dimensionality::<Ix3>()
            .context("expected logits with shape [batch, seq, labels]")?;
        let predictions = argmax_label_indices(logits);

        if show_tokens {
            for (index, token) in encoding.get_tokens().iter().enumerate() {
                let label = self
                    .labels
                    .get(predictions.get(index).copied().unwrap_or_default())
                    .map(String::as_str)
                    .unwrap_or("<unknown>");
                let (start, end) = encoding.get_offsets().get(index).copied().unwrap_or((0, 0));
                debug!("{index:>3}: {token:<20} {label:<32} [{start}..{end}]");
            }
        }

        let decode_start = Instant::now();
        let entities = decode_entities(
            text,
            encoding,
            &predictions,
            &self.labels,
            self.decode_strategy,
        );
        record_timing(timings_ms, "infer.decode", decode_start);

        Ok(entities)
    }

    /// Sliding-window inference: re-tokenize with truncation+stride, run each window
    /// through ONNX, stitch entities across windows.
    fn infer_windowed(
        &mut self,
        text: &str,
        show_tokens: bool,
        timings_ms: &mut BTreeMap<String, f64>,
    ) -> anyhow::Result<(Vec<EntitySpan>, usize)> {
        // Clone tokenizer and enable truncation with stride for overflow windows.
        let retokenize_start = Instant::now();
        let mut windowed_tokenizer = self.tokenizer.clone();
        windowed_tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: self.max_tokens,
                stride: self.overlap_tokens,
                strategy: TruncationStrategy::OnlyFirst,
                direction: TruncationDirection::Right,
            }))
            .map_err(|err| anyhow!("failed to configure windowed truncation: {err}"))?;

        let mut primary = windowed_tokenizer
            .encode(text, true)
            .map_err(|err| anyhow!("windowed tokenization failed: {err}"))?;
        let overflow = primary.take_overflowing();

        let mut encodings = Vec::with_capacity(1 + overflow.len());
        encodings.push(primary);
        encodings.extend(overflow);
        let window_count = encodings.len();
        record_timing(timings_ms, "infer.windowed_tokenize", retokenize_start);
        info!("sliding window: {window_count} windows");

        // Run each window through ONNX and collect entities with emit regions.
        let mut window_results = Vec::with_capacity(window_count);
        for (i, encoding) in encodings.iter().enumerate() {
            if show_tokens {
                debug!("--- window {i} ---");
            }

            let window_start = Instant::now();
            let entities = self.infer_single_encoding(text, encoding, show_tokens, timings_ms)?;
            record_timing(timings_ms, &format!("infer.window_{i}"), window_start);

            let (emit_start, emit_end) =
                compute_emit_region(text, encoding, i, window_count, self.overlap_tokens);
            info!(
                "window {i}: {} entities, emit region [{emit_start}..{emit_end})",
                entities.len()
            );

            window_results.push(WindowEntities {
                entities,
                emit_start,
                emit_end,
            });
        }

        // Stitch entities across windows.
        let stitch_start = Instant::now();
        let entities = crate::window::stitch(window_results);
        record_timing(timings_ms, "infer.stitch", stitch_start);
        info!("stitched to {} entities", entities.len());

        Ok((entities, window_count))
    }
}

/// Parse config.json's id2label map into a contiguous Vec<String> indexed by label id.
/// Rejects sparse maps (gaps would create silent decode bugs).
fn load_labels(config_path: &Path) -> anyhow::Result<Vec<String>> {
    let config_text = fs::read_to_string(config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config: Value = serde_json::from_str(&config_text)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;

    let id2label = config
        .get("id2label")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("config.json is missing id2label"))?;

    let mut labels = BTreeMap::new();
    for (key, value) in id2label {
        let index = key
            .parse::<usize>()
            .with_context(|| format!("invalid label index {key}"))?;
        let label = value
            .as_str()
            .ok_or_else(|| anyhow!("label {key} is not a string"))?;
        labels.insert(index, label.to_owned());
    }

    let max_index = labels
        .keys()
        .next_back()
        .copied()
        .ok_or_else(|| anyhow!("id2label is empty"))?;

    let mut ordered = vec![String::new(); max_index + 1];
    for (index, label) in labels {
        ordered[index] = label;
    }

    // Guard against sparse id2label maps: a gap would produce an empty-string label
    // that silently passes "O" checks in decode and creates garbage entities.
    if let Some(gap) = ordered.iter().position(|l| l.is_empty()) {
        bail!(
            "id2label has a gap at index {gap} — all indices from 0 to {max_index} must be present"
        );
    }

    Ok(ordered)
}

fn validate_model_bundle(model_dir: &Path) -> anyhow::Result<()> {
    let missing = REQUIRED_MODEL_FILES
        .iter()
        .map(|relative_path| model_dir.join(relative_path))
        .filter(|path| !path.is_file())
        .collect::<Vec<PathBuf>>();

    if missing.is_empty() {
        return Ok(());
    }

    let missing_list = missing
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");

    bail!(
        "model dir '{}' is missing required files: {}",
        model_dir.display(),
        missing_list
    );
}

/// Compute the byte range where a window's predictions are authoritative.
///
/// Uses the same symmetric overlap trimming as windowed-span: trim `overlap/2` tokens
/// from the left (except first window), `overlap - overlap/2` from the right (except last).
/// The emit region is defined by the byte offsets of the resulting usable token boundaries.
fn compute_emit_region(
    text: &str,
    encoding: &tokenizers::Encoding,
    window_index: usize,
    window_count: usize,
    overlap_tokens: usize,
) -> (usize, usize) {
    let offsets = encoding.get_offsets();
    let special_mask = encoding.get_special_tokens_mask();

    // Usable tokens: non-special, non-zero-length.
    let usable: Vec<(usize, usize)> = offsets
        .iter()
        .zip(special_mask.iter())
        .filter_map(|(&(start, end), &special)| {
            if special == 0 && end > start {
                Some((start, end))
            } else {
                None
            }
        })
        .collect();

    if usable.is_empty() {
        return (0, text.len());
    }

    let left_trim = if window_index == 0 {
        0
    } else {
        overlap_tokens / 2
    };
    let right_trim = if window_index + 1 == window_count {
        0
    } else {
        overlap_tokens - overlap_tokens / 2
    };

    let emit_start_idx = left_trim.min(usable.len().saturating_sub(1));
    let emit_end_idx = usable
        .len()
        .saturating_sub(right_trim)
        .max(emit_start_idx + 1)
        .min(usable.len());

    let emit_start = if window_index == 0 {
        0
    } else {
        usable[emit_start_idx].0
    };
    let emit_end = if window_index + 1 == window_count {
        text.len()
    } else {
        usable[emit_end_idx - 1].1
    };

    (emit_start, emit_end)
}

fn ensure_sequence_within_limit(
    profile_name: &str,
    seq_len: usize,
    max_tokens: usize,
) -> anyhow::Result<()> {
    if seq_len > max_tokens {
        bail!(
            "profile '{profile_name}' tokenized to {seq_len} tokens, exceeding max_tokens={max_tokens}. \
             Set overlap_tokens > 0 in the profile to enable sliding-window inference, or shorten the input."
        );
    }

    Ok(())
}

fn record_timing(timings_ms: &mut BTreeMap<String, f64>, stage: &str, started_at: Instant) {
    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1_000.0;
    timings_ms.insert(stage.to_owned(), elapsed_ms);
    info!(target: "timing", "{stage}: {elapsed_ms:.2} ms");
}

fn ort_error<E>(err: ort::Error<E>) -> anyhow::Error {
    anyhow!(err.to_string())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::{ensure_sequence_within_limit, validate_model_bundle};

    #[test]
    fn rejects_missing_required_model_files() {
        let temp_dir = tempdir().expect("temp dir");
        let err = validate_model_bundle(temp_dir.path()).expect_err("missing files should fail");
        let message = err.to_string();

        assert!(message.contains("tokenizer.json"));
        assert!(message.contains("config.json"));
        assert!(message.contains("onnx/model_quantized.onnx"));
    }

    #[test]
    fn accepts_valid_model_bundle_shape() {
        let temp_dir = tempdir().expect("temp dir");
        fs::write(temp_dir.path().join("tokenizer.json"), "{}").expect("tokenizer");
        fs::write(temp_dir.path().join("config.json"), "{}").expect("config");
        fs::create_dir_all(temp_dir.path().join("onnx")).expect("onnx dir");
        fs::write(temp_dir.path().join("onnx/model_quantized.onnx"), "").expect("onnx model");

        validate_model_bundle(temp_dir.path()).expect("bundle should validate");
    }

    #[test]
    fn rejects_sequence_lengths_over_profile_limit() {
        let err = ensure_sequence_within_limit("eu_pii", 513, 512)
            .expect_err("limit overflow should fail");

        let message = err.to_string();
        assert!(message.contains("exceeding max_tokens=512"));
        assert!(message.contains("overlap_tokens"));
    }
}
