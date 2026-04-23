use std::sync::LazyLock;

use regex::Regex;

use crate::decode::EntitySpan;

const DATE_TIME_LABEL: &str = "DATE_TIME";
const DATE_TIME_SCORE: f32 = 0.8;

static DATE_TIME_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(
            r"\b\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])[Tt ]([01]\d|2[0-3]):[0-5]\d(:[0-5]\d)?(\.\d+)?([+-]([01]\d|2[0-3]):?[0-5]\d|Z)?\b",
        )
        .expect("valid ISO datetime regex"),
        Regex::new(r"\b\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
            .expect("valid ISO date regex"),
        Regex::new(r"\b(0[1-9]|[12]\d|3[01])\.(0[1-9]|1[0-2])\.(19|20)\d{2}\b")
            .expect("valid dd.mm.yyyy regex"),
        Regex::new(
            r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(0?[1-9]|[12]\d|3[01])(st|nd|rd|th)?[,]?\s+(19|20)\d{2}\b",
        )
        .expect("valid English month date regex"),
        Regex::new(
            r"(?i)\b(0?[1-9]|[12]\d|3[01])\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?,?\s+(19|20)\d{2}\b",
        )
        .expect("valid English day month date regex"),
        Regex::new(
            r"(?i)\b(0?[1-9]|[12]\d|3[01])\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+de\s+(19|20)\d{2}\b",
        )
        .expect("valid Spanish date regex"),
        Regex::new(
            r"(?i)\b(0?[1-9]|[12]\d|3[01])\.\s*(januar|februar|märz|maerz|april|mai|juni|juli|august|september|oktober|november|dezember)\s+(19|20)\d{2}\b",
        )
        .expect("valid German date regex"),
    ]
});

pub(crate) fn detect(text: &str) -> Vec<EntitySpan> {
    let mut spans = Vec::new();
    for pattern in DATE_TIME_PATTERNS.iter() {
        for mat in pattern.find_iter(text) {
            spans.push((mat.start(), mat.end()));
        }
    }

    dedupe_overlaps(spans)
        .into_iter()
        .filter_map(|(start, end)| {
            text.get(start..end).map(|matched| EntitySpan {
                label: DATE_TIME_LABEL.to_owned(),
                start,
                end,
                text: matched.to_owned(),
                score: DATE_TIME_SCORE,
            })
        })
        .collect()
}

fn dedupe_overlaps(mut spans: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    spans.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then((b.1 - b.0).cmp(&(a.1 - a.0)))
            .then(a.1.cmp(&b.1))
    });

    let mut filtered: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    for span in spans {
        if let Some(last) = filtered.last_mut()
            && span.0 < last.1
        {
            let last_len = last.1 - last.0;
            let span_len = span.1 - span.0;
            if span_len > last_len || (span_len == last_len && span.1 > last.1) {
                *last = span;
            }
            continue;
        }
        filtered.push(span);
    }

    filtered
}

#[cfg(test)]
mod tests {
    use super::detect;

    fn matched_texts<'a>(text: &'a str, matches: &[(usize, usize)]) -> Vec<&'a str> {
        matches
            .iter()
            .map(|(start, end)| &text[*start..*end])
            .collect::<Vec<_>>()
    }

    #[test]
    fn detects_iso_datetime_and_date() {
        let text = "Start 2024-03-20T14:30:00Z end 2024-01-15.";
        let entities = detect(text);
        let spans = entities
            .iter()
            .map(|e| (e.start, e.end))
            .collect::<Vec<_>>();

        assert_eq!(
            matched_texts(text, &spans),
            vec!["2024-03-20T14:30:00Z", "2024-01-15"]
        );
        assert!(entities.iter().all(|entity| entity.label == "DATE_TIME"));
    }

    #[test]
    fn detects_european_numeric_date() {
        let text = "Termin am 15.01.2024 um 10 Uhr.";
        let entities = detect(text);

        assert_eq!(entities.len(), 1);
        assert_eq!(&text[entities[0].start..entities[0].end], "15.01.2024");
    }

    #[test]
    fn detects_month_name_dates_multilingual() {
        let text = "January 15, 2024 | 15 de enero de 2024 | 15. März 2024";
        let entities = detect(text);
        let spans = entities
            .iter()
            .map(|e| (e.start, e.end))
            .collect::<Vec<_>>();

        assert_eq!(
            matched_texts(text, &spans),
            vec!["January 15, 2024", "15 de enero de 2024", "15. März 2024"]
        );
    }

    #[test]
    fn skips_ambiguous_and_weak_patterns() {
        let text = "01/15/2024 15/01/2024 2024 12/2024";
        let entities = detect(text);

        assert!(entities.is_empty());
    }

    #[test]
    fn keeps_longer_overlap_when_patterns_collide() {
        let text = "Window 2024-03-20 14:30:00 done.";
        let entities = detect(text);

        assert_eq!(entities.len(), 1);
        assert_eq!(
            &text[entities[0].start..entities[0].end],
            "2024-03-20 14:30:00"
        );
    }
}
