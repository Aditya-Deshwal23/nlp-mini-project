# ECBias

ECBias studies how analyst questions and executive answers unfold in earnings-call Q&A. It uses
interpretable classical NLP, transparent quality scores, and cautious group comparisons. It does
not use paid APIs, GPUs, or LLM calls.

## Current result headline

The generated fixture run contains **12 calls**, **11 calls with a detected Q&A section**, and
**12 exchanges**. The current [summary](results/SUMMARY.md) says that no statistically tested
primary results are available; the existing outputs are descriptive scaffolding only. No claim
about bias is made.

## Quickstart

```bash
make setup
source .venv/bin/activate
```

Place licensed transcript files under `data/raw/` or use the included synthetic fixtures for a
demo. Then run:

```bash
ecbias run-all --sample 12 --config configs/dataset.yaml
streamlit run app/streamlit_app.py
```

For checks:

```bash
make check
```

## Repository layout

- `src/ecbias/io/`: transcript loaders and schemas
- `src/ecbias/parsing/`: cleaning, Q&A segmentation, and exchange construction
- `src/ecbias/people/`: roles, inferred perceived gender, and privacy-safe enrichment
- `src/ecbias/argmining/`: sentence segmentation, rules, annotation, and classifier
- `src/ecbias/features/`: AQS, response, interaction, and merged feature tables
- `src/ecbias/scoring/`: transparent Argument Quality Score
- `src/ecbias/stats/`: analysis outputs
- `src/ecbias/viz/`: figures, live analysis, and power simulation
- `configs/`: dataset, scoring, lexicon, and feature specifications
- `data/`: raw, intermediate, processed, and annotation data
- `results/`: generated tables, figures, summaries, and run manifests
- `app/`: offline Streamlit dashboard
- `tests/`: unit, integration, edge-case, and contract tests
- `docs/`: methods, ethics, testing, data dictionary, and diagrams

## Architecture and screenshots

The design is shown in [architecture.svg](docs/diagrams/architecture.svg), a worked exchange is
shown in [dataflow_exchange.svg](docs/diagrams/dataflow_exchange.svg), and the table joins are
shown in [schema.svg](docs/diagrams/schema.svg). Dashboard screenshots: **TODO** (not generated
by the code).

## Limitations

Gender is inferred perceived gender from first names, is binary in the primary analysis, and is
uncertain. AQS is a transparent proxy, not ground truth. The synthetic fixture corpus is small,
and the current result tables do not contain inferential estimates. The data are observational:
any differences would be associations, never proof of bias or proof that bias is absent.
Transcripts have no timestamps, so latency and talk time are not measured; word counts are only
an airtime proxy.

See [ETHICS_AND_LIMITATIONS.md](docs/ETHICS_AND_LIMITATIONS.md) and
[TESTING.md](docs/TESTING.md).

## Licence and data citation

Project code is released under the MIT Licence in [LICENSE](LICENSE). The included synthetic
fixtures are for testing only. For an external transcript dataset, record its provider, version,
URL, access date, and licence here before reporting results: **TODO: cite the licensed transcript
dataset used for coursework**.

## Known limitations and future work

- Build a larger hand-annotated gold set and include a second annotator.
- Add a finance-specific lexicon such as Loughran–McDonald, subject to its
  licence and an optional local download.
- Use timestamps when they are available so latency and talk-time measures can
  be studied.
- Use non-binary or self-reported gender if the data and consent allow it.
