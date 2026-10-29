# Testing

ECBias uses several test layers so a change can be checked at the smallest
useful scope:

- **Unit tests** in `tests/test_*.py` cover individual parsers, scorers,
  feature builders, and CLI behavior.
- **Integration tests** in `tests/integration/` run the synthetic fixture
  pipeline and check stage row counts and output contracts.
- **Edge tests** in `tests/edge/` exercise malformed, empty, unusual, and
  under-powered inputs.
- **Contract checks** compare generated feature columns with
  `configs/feature_spec.yaml` and validate documentation links.
- **Determinism checks** compare hashes from repeated fixture runs.

Run one layer with:

```bash
.venv/bin/pytest tests/test_features.py
.venv/bin/pytest tests/integration
.venv/bin/pytest tests/edge
```

Run the complete gate with:

```bash
make check
```

Known gaps include the small synthetic corpus, the absence of gold labels for
most real-world transcript formats, and the fact that offline dashboard smoke
tests do not replace browser-based visual review.
