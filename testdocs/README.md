# Test Fixtures

`testdocs/` now has two kinds of realistic synthetic inputs for built-in multilingual NER model.

Automated regression fixtures:

- `xenova_ner_windowed_*`: medium-size synthetic dossier that should force multi-window inference and verify stable placeholder reuse for person, org, and location entities.
- `xenova_ner_stress_windowed_*`: larger bundle that pushes repeated entity values across deep sliding-window inference.

These expected manifests focus on anonymization behavior:

- anonymized text contains expected placeholders
- repeated exact values reuse the same placeholder
- source literals used in the stable fixture set do not remain in anonymized output
- long inputs still produce multiple windows and stable total timing metadata

Exploratory manual probes:

- `xenova_ner_hard_probe_input.md`
- `xenova_ner_hard_stress_input.md`

The hard probes intentionally include entity shapes current model may miss, including emails, URLs, IPs, IDs, addresses, and bank data. They are useful for characterizing current model behavior, but they are not treated as stable pass/fail regression fixtures yet.
