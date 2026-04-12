// Entity stitching for sliding-window inference.
//
// When input text exceeds max_tokens, inference runs on overlapping windows.
// Each window produces entities with document-level byte offsets. This module
// filters entities by each window's emit region (midpoint test) and deduplicates
// same-label overlapping spans (longest wins).
//
// Emit-region computation happens in runtime.rs (where encodings are available).
// This module only needs the pre-computed emit boundaries.

use crate::decode::EntitySpan;

/// A single window's inference output with its emit region.
#[derive(Debug)]
pub struct WindowEntities {
    /// Decoded entities with document-level byte offsets.
    pub entities: Vec<EntitySpan>,
    /// Start of this window's authoritative emit region (byte offset, inclusive).
    pub emit_start: usize,
    /// End of this window's authoritative emit region (byte offset, exclusive).
    pub emit_end: usize,
}

/// Filter each window's entities by emit region, then deduplicate same-label overlaps.
///
/// An entity is "owned" by a window if its byte midpoint falls within the window's
/// emit region. After filtering, same-label spans that overlap in byte range are
/// resolved by keeping the longest span.
pub fn stitch(windows: Vec<WindowEntities>) -> Vec<EntitySpan> {
    let mut kept: Vec<EntitySpan> = Vec::new();

    for window in windows {
        for entity in window.entities {
            let midpoint = entity.start + (entity.end - entity.start) / 2;
            if midpoint >= window.emit_start && midpoint < window.emit_end {
                kept.push(entity);
            }
        }
    }

    deduplicate(kept)
}

/// Remove same-label overlapping spans, keeping the longest.
///
/// Two spans of the same label "overlap" if their byte ranges intersect.
/// Different labels are allowed to overlap (e.g. ORG and LOC can share bytes).
fn deduplicate(mut entities: Vec<EntitySpan>) -> Vec<EntitySpan> {
    if entities.len() <= 1 {
        return entities;
    }

    // Sort by label and byte offset to build overlap clusters deterministically.
    entities.sort_by(|a, b| {
        a.label
            .cmp(&b.label)
            .then(a.start.cmp(&b.start))
            .then(a.end.cmp(&b.end))
    });

    let mut result: Vec<EntitySpan> = Vec::new();
    let mut i = 0;

    while i < entities.len() {
        let label = entities[i].label.clone();
        let mut cluster_end = entities[i].end;
        let mut cluster_last = i + 1;

        // Build one transitive overlap cluster for this label.
        while cluster_last < entities.len()
            && entities[cluster_last].label == label
            && entities[cluster_last].start < cluster_end
        {
            cluster_end = cluster_end.max(entities[cluster_last].end);
            cluster_last += 1;
        }

        let winner = entities[i..cluster_last]
            .iter()
            .max_by(|a, b| {
                let a_len = a.end.saturating_sub(a.start);
                let b_len = b.end.saturating_sub(b.start);
                a_len
                    .cmp(&b_len)
                    .then_with(|| b.start.cmp(&a.start))
                    .then_with(|| b.end.cmp(&a.end))
            })
            .expect("cluster always contains at least one entity");
        result.push(winner.clone());
        i = cluster_last;
    }

    // Final sort by byte offset for stable output.
    result.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(a.end.cmp(&b.end))
            .then(a.label.cmp(&b.label))
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::EntitySpan;

    fn span(label: &str, start: usize, end: usize, text: &str) -> EntitySpan {
        EntitySpan {
            label: label.to_owned(),
            start,
            end,
            text: text.to_owned(),
        }
    }

    fn window(entities: Vec<EntitySpan>, emit_start: usize, emit_end: usize) -> WindowEntities {
        WindowEntities {
            entities,
            emit_start,
            emit_end,
        }
    }

    #[test]
    fn single_window_passthrough() {
        let entities = vec![span("PER", 0, 8, "John Doe"), span("LOC", 13, 18, "Paris")];
        let windows = vec![window(entities.clone(), 0, 100)];

        let result = stitch(windows);
        assert_eq!(result, entities);
    }

    #[test]
    fn entity_assigned_to_correct_window_by_midpoint() {
        // Entity at bytes 40..50, midpoint = 45.
        // Window 0 emits [0, 42), window 1 emits [42, 100).
        // Midpoint 45 is in window 1's region → window 1 owns it.
        let entity = span("PER", 40, 50, "some name");
        let w0 = window(vec![entity.clone()], 0, 42);
        let w1 = window(vec![entity.clone()], 42, 100);

        let result = stitch(vec![w0, w1]);
        assert_eq!(result, vec![entity]);
    }

    #[test]
    fn duplicate_entity_from_two_windows_deduplicated() {
        // Both windows detect the same entity; only one survives.
        let entity = span("ORG", 50, 66, "Microsoft Spain");
        let w0 = window(vec![entity.clone()], 0, 60);
        let w1 = window(vec![entity.clone()], 55, 120);

        let result = stitch(vec![w0, w1]);
        assert_eq!(result, vec![entity]);
    }

    #[test]
    fn overlapping_same_label_keeps_longest() {
        // Two PER spans overlap: one is longer.
        let short = span("PER", 10, 18, "John Doe");
        let long = span("PER", 10, 22, "John Doe III");
        let w0 = window(vec![short], 0, 50);
        let w1 = window(vec![long.clone()], 0, 50);

        let result = stitch(vec![w0, w1]);
        assert_eq!(result, vec![long]);
    }

    #[test]
    fn overlapping_same_label_keeps_longest_even_if_it_starts_later() {
        // Regression: the previous implementation kept earliest-starting span, not longest.
        let earlier_short = span("PER", 10, 16, "John D");
        let later_long = span("PER", 12, 24, "hn Doe Senior");
        let w0 = window(vec![earlier_short], 0, 50);
        let w1 = window(vec![later_long.clone()], 0, 50);

        let result = stitch(vec![w0, w1]);
        assert_eq!(result, vec![later_long]);
    }

    #[test]
    fn different_labels_can_overlap() {
        let org = span("ORG", 0, 10, "Apple Inc.");
        let loc = span("LOC", 6, 15, "Inc. Park");
        let w = window(vec![org.clone(), loc.clone()], 0, 100);

        let result = stitch(vec![w]);
        assert_eq!(result, vec![org, loc]);
    }

    #[test]
    fn output_sorted_by_byte_offset() {
        // Entities arrive in reverse order from different windows.
        let first = span("PER", 10, 18, "John Doe");
        let second = span("LOC", 50, 55, "Paris");
        let w0 = window(vec![second.clone()], 40, 100);
        let w1 = window(vec![first.clone()], 0, 40);

        let result = stitch(vec![w0, w1]);
        assert_eq!(result, vec![first, second]);
    }

    #[test]
    fn empty_windows_produce_no_entities() {
        let w0 = window(vec![], 0, 50);
        let w1 = window(vec![], 50, 100);

        let result = stitch(vec![w0, w1]);
        assert!(result.is_empty());
    }

    #[test]
    fn entity_outside_emit_region_is_dropped() {
        // Entity midpoint at 5, but emit region starts at 10.
        let entity = span("PER", 0, 10, "John Doe");
        let w = window(vec![entity], 10, 50);

        let result = stitch(vec![w]);
        assert!(result.is_empty());
    }
}
