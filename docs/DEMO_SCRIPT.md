# ECBias demo script: 9 minutes

This script uses the synthetic fixtures. It is a reproducibility demo, not an
empirical claim about real earnings calls.

## 0:00–1:00 — Problem and why it matters

**Page/command:** opening slide or README.

**Say:** “Analyst questions and executive answers are interactions, not isolated
sentences. I want to describe structure, response style, and possible gender
associations using transparent measures. The unit is one exchange: one analyst
question followed by executive answers. The current fixture run has 12 calls,
11 detected Q&A calls, and 12 exchanges.”

**Fallback:** open [data_quality.csv](../results/tables/data_quality.csv) and
read the same counts.

## 1:00–3:00 — Pipeline walkthrough

**Page/command:** open [architecture.svg](diagrams/architecture.svg), then run:

```bash
ecbias run-all --sample 12 --config configs/dataset.yaml
```

**Say:** “Loaders create canonical turns. Parsing cleans text, finds Q&A, and
builds exchanges. Enrichment adds roles and salted IDs. Sentence rules and
classical features produce labels, AQS, response measures, and interaction
measures. The final stages write tables, figures, and a run manifest.”

**Fallback:** use the pre-rendered files in `results/figures/` and
`results/run_manifest.json`; do not rerun the pipeline during the demo.

## 3:00–5:00 — Argument mining and live analysis

**Page:** Streamlit → **Live analyser**.

**Say:** “The live analyser applies the same sentence rules and scoring
functions used by the pipeline. REQUEST has priority for a question, while
numeric or evidence markers can trigger EVIDENCE. The labels are interpretable
signals, not truth judgments.”

Use the prefilled synthetic example and click **Analyse**. Point to the
sentence labels, component bars, response metrics, and timing readout.

**Fallback:** show the pre-rendered argument labels in the dashboard or open
`data/processed/sentences_labeled.parquet`.

## 5:00–7:00 — Results and forest plot

**Page:** Streamlit → **Bias results**, then open
`results/figures/04_primary_effect_forest.png`.

**Say:** “The project preregisters the primary outcomes and planned
multiple-testing control. In this current run, the primary test table is empty,
so there is no inferential forest plot to interpret. The honest conclusion is
that statistically tested primary results are not available yet—not that there
is no difference.”

**Fallback:** show the generated empty-result figure and
`results/SUMMARY.md`.

## 7:00–8:00 — Limitations and ethics

**Page:** [ETHICS_AND_LIMITATIONS.md](ETHICS_AND_LIMITATIONS.md).

**Say:** “Gender is inferred perceived gender from names, binary in the
primary analysis, and says nothing about identity. IDs are salted, results are
aggregate, and the data are observational. AQS is a proxy. The synthetic
fixture is deliberately small.”

**Fallback:** keep the limitations box visible on the Overview page.

## 8:00–9:00 — Questions

**Page:** [VIVA_QA.md](VIVA_QA.md).

**Say:** “I can explain every major feature, label priority, hypothesis, and
limitation. If a number is not in a generated result table, I will say TODO
rather than estimate it.”

**Fallback:** use the pre-rendered figures and this document if the dashboard
or model download is unavailable.

## Rehearsal record

The script is designed for a nine-minute run. A recorded timed rehearsal is
not stored in the repository: **TODO: record the student's final rehearsal
time before submission**.
