# Test Fixtures

`testdocs/` has two kinds of synthetic inputs for built-in multilingual NER profile.

## Stable regression fixtures

- `xenova_ner_windowed_*`: medium input that forces multi-window inference
- `xenova_ner_stress_windowed_*`: larger input that pushes repeated values across deeper sliding-window inference

These fixtures assert only:

- anonymized text contains expected placeholders
- repeated exact values reuse same placeholder
- source literals listed in fixture manifests do not remain in anonymized output
- long inputs still produce at least the expected minimum window count

Fixture placeholder ids and replacement counts are pinned to current built-in model/profile. Treat them as regression guards for this repo, not as cross-model guarantees.

## Exploratory probes

- `xenova_ner_hard_probe_input.md`
- `xenova_ner_hard_stress_input.md`

These probe entity shapes current model may miss, including emails, URLs, IPs, IDs, addresses, and bank data. Keep them for manual characterization; they are not stable pass/fail fixtures.
