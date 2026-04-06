# Test Fixtures

These fixtures are profile-aligned regression inputs for local model testing.

- `eu_pii_windowed_*`: long synthetic text centered on `eu_pii` entity types, with the manifest asserting only the stable subset we want to regress.
- `xenova_ner_windowed_*`: long synthetic text that stays within the narrower `PER` / `ORG` / `LOC` coverage of `xenova_ner_hrl`.

The expected manifests are intentionally partial:

- they check repeated exact entities we expect the model to find
- they assert the long input actually uses multiple windows
- they do not try to assert full extraction exhaustiveness or anonymized-text output

Run them with:

```bash
just test-fixtures
```
