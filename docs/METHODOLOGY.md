# Methodology

## Argument mining

spaCy segments question and answer text into sentences. The rule labeller retains boolean flags
and assigns one primary label using this priority:

`REQUEST > REBUTTAL > EVIDENCE > CLAIM > OTHER`.

- **CLAIM:** an assertion or stance needing support.
- **EVIDENCE:** a fact, number, example, or source offered as support.
- **REBUTTAL:** pushback, contrast, or counter-point.
- **REQUEST:** an explicit ask or question.
- **OTHER:** procedure, pleasantries, or filler.

Rule-generated labels are **silver labels**. A hand-annotated file is the **gold set** used for
evaluation. The classifier uses TF-IDF word features, optional character n-grams, and hand-built
linguistic features. See [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md).

## Argument Quality Score

AQS components are percentile-scaled over the corpus and combined as:

`AQS = 100 × Σ(weight_component × component_score)`.

Question weights from [aqs.yaml](../configs/aqs.yaml):

- structure: 0.30
- specificity: 0.25
- clarity: 0.20
- calibration: 0.10
- focus: 0.15

Answer weights:

- structure: 0.35
- specificity: 0.25
- clarity: 0.20
- calibration: 0.20

Structure uses claims, evidence, and claim-support links. Specificity uses numeric and named
entity density. Clarity uses readability proximity to the configured band and a long-sentence
penalty. Calibration penalises hedge rates above the configured threshold. Focus is
`1 / max(number_of_requests, 1)`. AQS is a proxy, not ground truth.

## Statistical plan

The pre-specified plan is in [HYPOTHESES.md](HYPOTHESES.md). The unit is the exchange. Primary
comparisons use perceived gender groups F and M, with Unknown excluded from primary comparisons
and reported as coverage. Outcomes are AQS_q, request count, relevance, answer/question ratio,
answer hedge and deflection rates, courtesy opener, and AQS_a.

The intended analysis uses Welch and Mann–Whitney tests, standardised effect sizes with bootstrap
intervals, within-call gender permutations, binary-outcome tests, call-clustered OLS, call random
intercepts, within-call paired comparisons, and power calculations. Calls are the clustering unit
because exchanges from the same call are not independent. Benjamini-Hochberg adjustment is applied
to the primary family and separately to exploratory outcomes. Current inferential result tables
are empty, so numerical test conclusions are **TODO**.
