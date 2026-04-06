// Post-processing: turns raw ONNX logits into merged entity spans.
//
// Two decode strategies exist because different models tag differently:
//   - generic_bio: strict BIO tagging (B-PER, I-PER). Standard NER models.
//   - pii_relaxed: ignores BIO prefixes, merges across gaps for emails/phones.
//     Includes model-specific heuristics for the eu-pii model.
//
// Adding a new model that doesn't fit either strategy? Add a new DecodeStrategy variant
// and a corresponding merge function.

use std::fmt;

use ndarray::ArrayView3;
use serde::{Deserialize, Serialize};
use tokenizers::Encoding;

/// A detected entity: label (e.g. "PER"), byte offsets into the original text, and the text itself.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EntitySpan {
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub text: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecodeStrategy {
    GenericBio,
    PiiRelaxed,
}

impl Default for DecodeStrategy {
    fn default() -> Self {
        Self::GenericBio
    }
}

impl fmt::Display for DecodeStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::GenericBio => "generic_bio",
            Self::PiiRelaxed => "pii_relaxed",
        };
        f.write_str(value)
    }
}

/// Pick the highest-scoring label for each token position. Returns label indices.
pub fn argmax_label_indices(logits: ArrayView3<'_, f32>) -> Vec<usize> {
    let seq_len = logits.shape()[1];
    let label_count = logits.shape()[2];
    let mut predictions = Vec::with_capacity(seq_len);

    for token_index in 0..seq_len {
        let mut best_label = 0usize;
        let mut best_score = f32::NEG_INFINITY;

        for label_index in 0..label_count {
            let score = logits[(0, token_index, label_index)];
            if score > best_score {
                best_score = score;
                best_label = label_index;
            }
        }

        predictions.push(best_label);
    }

    predictions
}

/// Walk token predictions and merge adjacent tokens into entity spans.
/// Skips special tokens ([CLS]/[SEP]) and "O" labels. Merge behavior depends on strategy.
pub fn decode_entities(
    text: &str,
    encoding: &Encoding,
    predictions: &[usize],
    labels: &[String],
    strategy: DecodeStrategy,
) -> Vec<EntitySpan> {
    let special_mask = encoding.get_special_tokens_mask();
    let offsets = encoding.get_offsets();
    let upper = predictions.len().min(offsets.len()).min(special_mask.len());

    let mut entities = Vec::new();
    let mut current: Option<EntitySpan> = None;

    for index in 0..upper {
        if special_mask[index] != 0 {
            flush_entity(&mut current, &mut entities);
            continue;
        }

        let (start, end) = offsets[index];
        if start >= end || end > text.len() || text.get(start..end).is_none() {
            flush_entity(&mut current, &mut entities);
            continue;
        }

        let label = labels
            .get(predictions[index])
            .map(String::as_str)
            .unwrap_or("O");

        if label == "O" {
            flush_entity(&mut current, &mut entities);
            continue;
        }

        let (prefix, kind) = split_label(label);
        if kind == "O" {
            flush_entity(&mut current, &mut entities);
            continue;
        }

        let merge_label = current
            .as_ref()
            .and_then(|entity| {
                let gap_start = entity.end.min(start);
                let gap = text.get(gap_start..start).unwrap_or("");
                merged_label(strategy, entity.label.as_str(), kind, prefix, gap)
            })
            .map(str::to_owned);

        if current.is_none() || merge_label.is_none() {
            flush_entity(&mut current, &mut entities);
            current = Some(EntitySpan {
                label: kind.to_owned(),
                start,
                end,
                text: text.get(start..end).unwrap_or("").to_owned(),
            });
            continue;
        }

        if let Some(entity) = current.as_mut() {
            if let Some(merged) = merge_label {
                entity.label = merged;
            }
            entity.end = end;
            if let Some(span_text) = text.get(entity.start..entity.end) {
                entity.text = span_text.to_owned();
            }
        }
    }

    flush_entity(&mut current, &mut entities);
    entities
}

/// Splits a BIO-style label like "B-PER" into ("B", "PER").
/// Some models (e.g. eu-pii) emit bare labels like "EMAIL_ADDRESS" without a BIO prefix.
/// Those are treated as a B-tag (begin) so they start a new entity span.
fn split_label(label: &str) -> (&str, &str) {
    match label.split_once('-') {
        Some((prefix, kind)) if !kind.is_empty() => (prefix, kind),
        _ => ("B", label),
    }
}

fn flush_entity(current: &mut Option<EntitySpan>, entities: &mut Vec<EntitySpan>) {
    if let Some(entity) = current.take() {
        entities.push(entity);
    }
}

/// Decide if the next token should merge into the current entity span.
/// Returns Some(label) if yes, None if the token starts a new span.
fn merged_label<'a>(
    strategy: DecodeStrategy,
    current: &'a str,
    next: &'a str,
    prefix: &str,
    gap: &str,
) -> Option<&'a str> {
    match strategy {
        DecodeStrategy::GenericBio => merged_label_generic(current, next, prefix, gap),
        DecodeStrategy::PiiRelaxed => merged_label_pii_relaxed(current, next, gap),
    }
}

/// GenericBio: only merge if previous is B-X, next is I-X (same kind), with whitespace gap.
fn merged_label_generic<'a>(
    current: &'a str,
    next: &'a str,
    prefix: &str,
    gap: &str,
) -> Option<&'a str> {
    if current == next && prefix == "I" && is_generic_gap(gap) {
        return Some(current);
    }
    None
}

/// PiiRelaxed: ignores BIO prefix, merges same-kind tokens across type-specific gaps
/// (e.g. emails allow `.@_-+`, phones allow `+-()./` and whitespace).
fn merged_label_pii_relaxed<'a>(current: &'a str, next: &'a str, gap: &str) -> Option<&'a str> {
    if current == next {
        let can_merge = match current {
            "EMAIL_ADDRESS" => gap.is_empty() || gap.chars().all(|ch| "._-+@".contains(ch)),
            "PHONE_NUMBER" => gap
                .chars()
                .all(|ch| ch.is_whitespace() || "+-()./".contains(ch)),
            _ => is_generic_gap(gap),
        };

        if can_merge {
            return Some(current);
        }
    }

    // Model-specific heuristic: the eu-pii model frequently classifies the domain part of
    // an email (e.g. "@company.com") as ORGANIZATION_NAME. When this immediately follows an
    // EMAIL_ADDRESS token, absorb it into the email span.
    if current == "EMAIL_ADDRESS" && next == "ORGANIZATION_NAME" && gap.is_empty() {
        return Some("EMAIL_ADDRESS");
    }

    None
}

fn is_generic_gap(gap: &str) -> bool {
    gap.is_empty() || gap.chars().all(char::is_whitespace)
}

#[cfg(test)]
mod tests {
    use tokenizers::Encoding;

    use super::{DecodeStrategy, EntitySpan, decode_entities};

    fn test_encoding(tokens: &[&str], offsets: &[(usize, usize)]) -> Encoding {
        test_encoding_with_mask(tokens, offsets, &vec![0; tokens.len()])
    }

    fn test_encoding_with_mask(
        tokens: &[&str],
        offsets: &[(usize, usize)],
        special_tokens_mask: &[u32],
    ) -> Encoding {
        let len = tokens.len();
        Encoding::new(
            vec![0; len],
            vec![0; len],
            tokens.iter().map(|token| (*token).to_owned()).collect(),
            vec![None; len],
            offsets.to_vec(),
            special_tokens_mask.to_vec(),
            vec![1; len],
            vec![],
            Default::default(),
        )
    }

    fn span(label: &str, start: usize, end: usize, text: &str) -> EntitySpan {
        EntitySpan {
            label: label.to_owned(),
            start,
            end,
            text: text.to_owned(),
        }
    }

    #[test]
    fn merges_generic_bio_spans() {
        let text = "John Doe";
        let encoding = test_encoding(&["John", "Doe"], &[(0, 4), (5, 8)]);
        let predictions = vec![1, 2];
        let labels = vec!["O", "B-PER", "I-PER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(
            text,
            &encoding,
            &predictions,
            &labels,
            DecodeStrategy::GenericBio,
        );

        assert_eq!(entities, vec![span("PER", 0, 8, "John Doe")]);
    }

    #[test]
    fn merges_relaxed_email_tokens() {
        let text = "alice@example.com";
        let encoding = test_encoding(&["alice", "@example", ".com"], &[(0, 5), (5, 13), (13, 17)]);
        let predictions = vec![1, 2, 2];
        let labels = vec!["O", "B-EMAIL_ADDRESS", "I-EMAIL_ADDRESS"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(
            text,
            &encoding,
            &predictions,
            &labels,
            DecodeStrategy::PiiRelaxed,
        );

        assert_eq!(
            entities,
            vec![span("EMAIL_ADDRESS", 0, 17, "alice@example.com")]
        );
    }

    #[test]
    fn merges_relaxed_phone_tokens() {
        let text = "+39 347 123 4567";
        let encoding = test_encoding(
            &["+39", "347", "123", "4567"],
            &[(0, 3), (4, 7), (8, 11), (12, 16)],
        );
        let predictions = vec![1, 2, 2, 2];
        let labels = vec!["O", "B-PHONE_NUMBER", "I-PHONE_NUMBER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(
            text,
            &encoding,
            &predictions,
            &labels,
            DecodeStrategy::PiiRelaxed,
        );

        assert_eq!(
            entities,
            vec![span("PHONE_NUMBER", 0, 16, "+39 347 123 4567")]
        );
    }

    #[test]
    fn does_not_merge_incompatible_labels() {
        let text = "John Paris";
        let encoding = test_encoding(&["John", "Paris"], &[(0, 4), (5, 10)]);
        let predictions = vec![1, 2];
        let labels = vec!["O", "B-PER", "I-LOC"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(
            text,
            &encoding,
            &predictions,
            &labels,
            DecodeStrategy::GenericBio,
        );

        assert_eq!(
            entities,
            vec![span("PER", 0, 4, "John"), span("LOC", 5, 10, "Paris")]
        );
    }

    #[test]
    fn skips_special_tokens() {
        let text = "John";
        // Simulates [CLS] John [SEP] — special tokens at positions 0 and 2.
        let encoding = test_encoding_with_mask(
            &["[CLS]", "John", "[SEP]"],
            &[(0, 0), (0, 4), (0, 0)],
            &[1, 0, 1],
        );
        let predictions = vec![1, 1, 1];
        let labels = vec!["O", "B-PER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(
            text,
            &encoding,
            &predictions,
            &labels,
            DecodeStrategy::GenericBio,
        );

        assert_eq!(entities, vec![span("PER", 0, 4, "John")]);
    }

    #[test]
    fn handles_bare_labels_without_bio_prefix() {
        // The eu-pii model emits bare labels like "EMAIL_ADDRESS" (no B-/I- prefix).
        // split_label treats these as B-tags, starting a new entity span.
        let text = "alice@example.com";
        let encoding = test_encoding(&["alice", "@example", ".com"], &[(0, 5), (5, 13), (13, 17)]);
        let predictions = vec![1, 1, 1];
        let labels = vec!["O", "EMAIL_ADDRESS"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(
            text,
            &encoding,
            &predictions,
            &labels,
            DecodeStrategy::PiiRelaxed,
        );

        assert_eq!(
            entities,
            vec![span("EMAIL_ADDRESS", 0, 17, "alice@example.com")]
        );
    }
}
