// Post-processing: turns raw ONNX logits into merged entity spans for the
// built-in Xenova DistilBERT NER model. Merge rules stay mostly strict BIO,
// with conservative repair for obvious malformed subword runs.

use std::borrow::Cow;

use ndarray::ArrayView3;
use serde::Serialize;
use tokenizers::Encoding;

/// A detected entity: label (e.g. "PER"), byte offsets into the original text, and the text itself.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EntitySpan {
    pub label: Cow<'static, str>,
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub score: f32,
}

struct DecodeView<'a> {
    text: &'a str,
    tokens: &'a [String],
    offsets: &'a [(usize, usize)],
    special_mask: &'a [u32],
    predictions_with_probs: &'a [(usize, f32)],
    labels: &'a [String],
    upper: usize,
}

/// Pick the highest-scoring label for each token position, returning both the
/// argmax label index and the softmax probability of that label.
///
/// Uses numerically stable softmax (subtract max). Assumes batch=0.
pub fn argmax_label_indices_with_probs(logits: ArrayView3<'_, f32>) -> Vec<(usize, f32)> {
    let seq_len = logits.shape()[1];
    let label_count = logits.shape()[2];
    let mut predictions = Vec::with_capacity(seq_len);

    for token_index in 0..seq_len {
        let mut best_label = 0usize;
        let mut best_logit = f32::NEG_INFINITY;

        for label_index in 0..label_count {
            let score = logits[(0, token_index, label_index)];
            if score > best_logit {
                best_logit = score;
                best_label = label_index;
            }
        }

        let mut sum_exp = 0.0f32;
        for label_index in 0..label_count {
            sum_exp += (logits[(0, token_index, label_index)] - best_logit).exp();
        }
        let prob = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

        predictions.push((best_label, prob));
    }

    predictions
}

/// Walk token predictions and merge adjacent tokens into entity spans.
/// Skips special tokens ([CLS]/[SEP]), invalid offsets, and non-BIO labels.
/// Also repairs two obvious malformed BIO cases:
/// - orphan `I-*` continuation subwords
/// - single-word `O` glitches inside an otherwise continuous same-family span
pub fn decode_entities(
    text: &str,
    encoding: &Encoding,
    predictions_with_probs: &[(usize, f32)],
    labels: &[String],
) -> Vec<EntitySpan> {
    let special_mask = encoding.get_special_tokens_mask();
    let offsets = encoding.get_offsets();
    let tokens = encoding.get_tokens();
    let upper = predictions_with_probs
        .len()
        .min(offsets.len())
        .min(special_mask.len())
        .min(tokens.len());
    let view = DecodeView {
        text,
        tokens,
        offsets,
        special_mask,
        predictions_with_probs,
        labels,
        upper,
    };

    let mut entities = Vec::new();
    let mut current: Option<EntitySpan> = None;
    let mut index = 0usize;

    while index < view.upper {
        if view.special_mask[index] != 0 {
            flush_entity(&mut current, &mut entities);
            index += 1;
            continue;
        }

        let Some((start, end)) = token_bounds(view.text, view.offsets, index) else {
            flush_entity(&mut current, &mut entities);
            index += 1;
            continue;
        };

        let (label_idx, token_prob) = view.predictions_with_probs[index];
        let label = view
            .labels
            .get(label_idx)
            .map(String::as_str)
            .unwrap_or("O");

        if label == "O" {
            if let Some(entity) = current.as_mut()
                && let Some((resume_index, repaired_end)) =
                    repair_intra_word_o_glitch(&view, entity, index)
            {
                entity.end = repaired_end;
                if let Some(span_text) = view.text.get(entity.start..entity.end) {
                    entity.text = span_text.to_owned();
                }
                index = resume_index;
                continue;
            }
            flush_entity(&mut current, &mut entities);
            index += 1;
            continue;
        }

        let Some((prefix, kind)) = split_bio_label(label) else {
            flush_entity(&mut current, &mut entities);
            index += 1;
            continue;
        };

        let merge_label = current
            .as_ref()
            .and_then(|entity| {
                let gap_start = entity.end.min(start);
                let gap = view.text.get(gap_start..start).unwrap_or("");
                merged_label(entity.label.as_ref(), kind, prefix, gap)
            })
            .map(str::to_owned);

        if current.is_none() || merge_label.is_none() {
            let repaired_start =
                if prefix == "I" && is_continuation_subword(view.tokens[index].as_str()) {
                    backfill_orphan_i_start(&view, index).unwrap_or(start)
                } else {
                    start
                };
            flush_entity(&mut current, &mut entities);
            current = Some(EntitySpan {
                label: Cow::Owned(kind.to_owned()),
                start: repaired_start,
                end,
                text: view.text.get(repaired_start..end).unwrap_or("").to_owned(),
                score: token_prob,
            });
            index += 1;
            continue;
        }

        if let Some(entity) = current.as_mut() {
            if let Some(merged) = merge_label {
                entity.label = Cow::Owned(merged);
            }
            entity.end = end;
            if let Some(span_text) = view.text.get(entity.start..entity.end) {
                entity.text = span_text.to_owned();
            }
            entity.score = entity.score.min(token_prob);
        }

        index += 1;
    }

    flush_entity(&mut current, &mut entities);
    entities
}

/// Splits a strict BIO label like "B-PER" into ("B", "PER").
/// Any bare or malformed label is ignored by decode_entities.
fn split_bio_label(label: &str) -> Option<(&str, &str)> {
    let (prefix, kind) = label.split_once('-')?;
    if kind.is_empty() {
        return None;
    }
    if prefix == "B" || prefix == "I" {
        return Some((prefix, kind));
    }
    None
}

fn flush_entity(current: &mut Option<EntitySpan>, entities: &mut Vec<EntitySpan>) {
    if let Some(entity) = current.take() {
        entities.push(entity);
    }
}

fn token_bounds(text: &str, offsets: &[(usize, usize)], index: usize) -> Option<(usize, usize)> {
    let (start, end) = offsets.get(index).copied()?;
    if start >= end || end > text.len() || text.get(start..end).is_none() {
        return None;
    }
    Some((start, end))
}

fn token_label<'a>(
    predictions_with_probs: &[(usize, f32)],
    labels: &'a [String],
    index: usize,
) -> &'a str {
    let (label_idx, _) = predictions_with_probs
        .get(index)
        .copied()
        .unwrap_or((0, 0.0));
    labels.get(label_idx).map(String::as_str).unwrap_or("O")
}

fn is_continuation_subword(token: &str) -> bool {
    token.starts_with("##")
}

fn repair_intra_word_o_glitch(
    view: &DecodeView<'_>,
    current: &EntitySpan,
    index: usize,
) -> Option<(usize, usize)> {
    if !is_continuation_subword(view.tokens.get(index)?.as_str()) {
        return None;
    }

    let (start, end) = token_bounds(view.text, view.offsets, index)?;
    if start != current.end || token_label(view.predictions_with_probs, view.labels, index) != "O" {
        return None;
    }

    let mut run_end = end;
    let mut scan = index;
    while scan < view.upper {
        if view.special_mask[scan] != 0 {
            break;
        }

        let Some((scan_start, scan_end)) = token_bounds(view.text, view.offsets, scan) else {
            break;
        };
        if scan_start != run_end.min(scan_start)
            || scan_start != if scan == index { current.end } else { run_end }
        {
            break;
        }

        if token_label(view.predictions_with_probs, view.labels, scan) != "O"
            || !is_continuation_subword(view.tokens[scan].as_str())
        {
            break;
        }

        run_end = scan_end;
        scan += 1;
    }

    if scan >= view.upper || view.special_mask[scan] != 0 {
        return None;
    }

    let (next_start, _) = token_bounds(view.text, view.offsets, scan)?;
    if next_start != run_end {
        return None;
    }

    let next_label = token_label(view.predictions_with_probs, view.labels, scan);
    let (prefix, kind) = split_bio_label(next_label)?;
    if prefix != "I" || kind != current.label.as_ref() {
        return None;
    }

    Some((scan, run_end))
}

fn backfill_orphan_i_start(view: &DecodeView<'_>, index: usize) -> Option<usize> {
    let (mut start, _) = token_bounds(view.text, view.offsets, index)?;
    let mut left = index;

    while left > 0 {
        let prev = left - 1;
        if view.special_mask[prev] != 0 {
            break;
        }

        let Some((prev_start, prev_end)) = token_bounds(view.text, view.offsets, prev) else {
            break;
        };
        if prev_end != start || token_label(view.predictions_with_probs, view.labels, prev) != "O" {
            break;
        }

        if !is_continuation_subword(view.tokens[prev].as_str())
            && !is_continuation_subword(view.tokens[left].as_str())
        {
            break;
        }

        start = prev_start;
        left = prev;
    }

    Some(start)
}

/// Decide if the next token should merge into the current entity span.
/// Returns Some(label) if yes, None if token starts a new span.
fn merged_label<'a>(current: &'a str, next: &'a str, prefix: &str, gap: &str) -> Option<&'a str> {
    if current != next {
        return None;
    }

    if gap.is_empty() || (prefix == "I" && is_generic_gap(gap)) {
        return Some(current);
    }

    None
}

fn is_generic_gap(gap: &str) -> bool {
    gap.is_empty() || gap.chars().all(char::is_whitespace)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use tokenizers::Encoding;

    use super::{EntitySpan, decode_entities};

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
            label: Cow::Owned(label.to_owned()),
            start,
            end,
            text: text.to_owned(),
            score: 1.0,
        }
    }

    fn span_with_score(
        label: &str,
        start: usize,
        end: usize,
        text: &str,
        score: f32,
    ) -> EntitySpan {
        EntitySpan {
            label: Cow::Owned(label.to_owned()),
            start,
            end,
            text: text.to_owned(),
            score,
        }
    }

    #[test]
    fn merges_bio_prefixed_spans() {
        let text = "John Doe";
        let encoding = test_encoding(&["John", "Doe"], &[(0, 4), (5, 8)]);
        let predictions = vec![(1, 1.0), (2, 1.0)];
        let labels = vec!["O", "B-PER", "I-PER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(entities, vec![span("PER", 0, 8, "John Doe")]);
    }

    #[test]
    fn merges_zero_gap_subword_fragments() {
        let text = "Microsoft";
        let encoding = test_encoding(&["Micro", "soft"], &[(0, 5), (5, 9)]);
        let predictions = vec![(1, 1.0), (2, 1.0)];
        let labels = vec!["O", "B-ORG", "I-ORG"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(entities, vec![span("ORG", 0, 9, "Microsoft")]);
    }

    #[test]
    fn merges_org_bio_continuation_with_whitespace() {
        let text = "Open AI";
        let encoding = test_encoding(&["Open", "AI"], &[(0, 4), (5, 7)]);
        let predictions = vec![(1, 1.0), (2, 1.0)];
        let labels = vec!["O", "B-ORG", "I-ORG"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(entities, vec![span("ORG", 0, 7, "Open AI")]);
    }

    #[test]
    fn merges_loc_bio_continuation_with_whitespace() {
        let text = "New York";
        let encoding = test_encoding(&["New", "York"], &[(0, 3), (4, 8)]);
        let predictions = vec![(1, 1.0), (2, 1.0)];
        let labels = vec!["O", "B-LOC", "I-LOC"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(entities, vec![span("LOC", 0, 8, "New York")]);
    }

    #[test]
    fn does_not_merge_incompatible_labels() {
        let text = "John Paris";
        let encoding = test_encoding(&["John", "Paris"], &[(0, 4), (5, 10)]);
        let predictions = vec![(1, 1.0), (2, 1.0)];
        let labels = vec!["O", "B-PER", "I-LOC"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(
            entities,
            vec![span("PER", 0, 4, "John"), span("LOC", 5, 10, "Paris")]
        );
    }

    #[test]
    fn skips_special_tokens() {
        let text = "John";
        let encoding = test_encoding_with_mask(
            &["[CLS]", "John", "[SEP]"],
            &[(0, 0), (0, 4), (0, 0)],
            &[1, 0, 1],
        );
        let predictions = vec![(1, 1.0), (1, 1.0), (1, 1.0)];
        let labels = vec!["O", "B-PER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(entities, vec![span("PER", 0, 4, "John")]);
    }

    #[test]
    fn ignores_bare_labels_without_bio_prefix() {
        let text = "alice";
        let encoding = test_encoding(&["alice"], &[(0, 5)]);
        let predictions = vec![(1, 1.0)];
        let labels = vec!["O", "PER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert!(entities.is_empty());
    }

    #[test]
    fn span_score_is_minimum_token_probability() {
        let text = "John Doe";
        let encoding = test_encoding(&["John", "Doe"], &[(0, 4), (5, 8)]);
        let predictions = vec![(1, 0.9), (2, 0.6)];
        let labels = vec!["O", "B-PER", "I-PER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(entities.len(), 1);
        assert!((entities[0].score - 0.6).abs() < 1e-6);
    }

    #[test]
    fn repairs_intra_word_o_glitch_across_same_entity_subwords() {
        let text = "Máximo Décimo Meridio";
        let encoding = test_encoding(
            &["Má", "##ximo", "D", "##éc", "##imo", "Mer", "##idi", "##o"],
            &[
                (0, 3),
                (3, 7),
                (8, 9),
                (9, 12),
                (12, 15),
                (16, 19),
                (19, 22),
                (22, 23),
            ],
        );
        let predictions = vec![
            (1, 0.9),
            (2, 0.8),
            (2, 0.7),
            (0, 0.01),
            (2, 0.6),
            (2, 0.95),
            (2, 0.97),
            (2, 0.99),
        ];
        let labels = vec!["O", "B-PER", "I-PER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(
            entities,
            vec![span_with_score("PER", 0, 23, "Máximo Décimo Meridio", 0.6)]
        );
    }

    #[test]
    fn backfills_orphan_i_continuation_to_same_word_start() {
        let text = "Décimo";
        let encoding = test_encoding(&["D", "##éc", "##imo"], &[(0, 1), (1, 4), (4, 7)]);
        let predictions = vec![(0, 1.0), (0, 1.0), (2, 0.8)];
        let labels = vec!["O", "B-PER", "I-PER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(entities, vec![span_with_score("PER", 0, 7, "Décimo", 0.8)]);
    }

    #[test]
    fn does_not_bridge_o_gap_across_whitespace() {
        let text = "John X Doe";
        let encoding = test_encoding(&["John", "X", "Doe"], &[(0, 4), (5, 6), (7, 10)]);
        let predictions = vec![(1, 0.9), (0, 1.0), (2, 0.8)];
        let labels = vec!["O", "B-PER", "I-PER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(
            entities,
            vec![
                span_with_score("PER", 0, 4, "John", 0.9),
                span_with_score("PER", 7, 10, "Doe", 0.8),
            ]
        );
    }

    #[test]
    fn orphan_i_backfill_stops_at_non_wordpiece_boundary() {
        let text = "ADécimo";
        let encoding = test_encoding(
            &["A", "D", "##éc", "##imo"],
            &[(0, 1), (1, 2), (2, 5), (5, 8)],
        );
        let predictions = vec![(0, 1.0), (0, 1.0), (0, 1.0), (2, 0.8)];
        let labels = vec!["O", "B-PER", "I-PER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(entities, vec![span_with_score("PER", 1, 8, "Décimo", 0.8)]);
    }
}
