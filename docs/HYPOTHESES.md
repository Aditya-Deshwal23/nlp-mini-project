# Hypotheses and Analysis Plan

This document is written before running the gender-difference analyses. It
defines the confirmatory outcomes and hypotheses in advance. Results may be
negative and will be reported plainly.

## Unit and primary population

The unit of analysis is the **exchange**: one analyst question followed by the
executive answer or answers that follow it.

The primary population is exchanges with an analyst whose inferred perceived
gender is **F or M**, and whose executive responders have a known perceived
gender. Gender is inferred probabilistically from names and is not a measure of
identity. Exchanges with `Unknown` gender are excluded from the primary
comparisons, but coverage and exclusions will be reported.

The data are observational. Any differences are associations, not proof of
bias or evidence that bias is absent.

## Primary outcomes and multiplicity

The following outcomes form one primary family and will be tested together
with Benjamini-Hochberg false-discovery-rate control:

- `AQS_q`: transparent question Argument Quality Score.
- `n_requests`: number of request-labelled sentences in the question.
- `relevance_tfidf`: TF-IDF cosine similarity between question and answer.
- `answer_to_question_ratio`: answer word count divided by question word count.
- `hedge_rate`: hedge-marker rate in the answer.
- `deflection_rate`: deflection-marker rate in the answer.
- `courtesy_opener`: whether the answer begins with a courtesy phrase.
- `AQS_a`: transparent answer Argument Quality Score.

The AQS measures are rubric-based proxies, not ground truth. Missing values
will not be silently imputed; analyses will use the available observations for
the variables involved and report the resulting sample sizes.

## Confirmatory hypotheses

All hypotheses are two-sided and use alpha = 0.05. Adjusted p-values will be
reported for the primary outcome family.

- **H1:** Question quality (`AQS_q`) differs by analyst perceived gender.
- **H2:** Answer relevance (`relevance_tfidf`) differs by analyst perceived
  gender.
- **H3:** Answer length ratio (`answer_to_question_ratio`) differs by analyst
  perceived gender.
- **H4:** Answer hedging (`hedge_rate`) and deflection (`deflection_rate`)
  differ by analyst perceived gender.
- **H5:** The probability of a courtesy opener (`courtesy_opener`) differs by
  analyst perceived gender.
- **H6:** Answer quality (`AQS_a`) differs by executive perceived gender.
- **H7:** There is an analyst perceived gender by executive perceived gender
  interaction on answer quality (`AQS_a`).

H4 contains two outcomes from the primary family. H7 is an interaction
analysis and will report the four analyst-gender by executive-gender cell
counts. It will be interpreted cautiously when cells are small.

## Exploratory analyses

Everything outside the primary outcome family and hypotheses above is
exploratory. Exploratory results will be labelled as such and will receive
separate multiple-testing control from the primary family. Exploratory analyses
include additional descriptive comparisons, model covariates, alternative AQS
scorers, rule-based versus ML argument labels, company-level clustering,
probable-gender exclusions, interrupted-exchange exclusions, and other
robustness checks.

## Reporting commitments

For each comparison, the report will show group sizes, estimates, uncertainty
intervals, effect sizes, and adjusted p-values where applicable. A result with
an adjusted p-value of at least 0.05 will be described as **“no statistically
detectable difference”**, not as proof of equality.

The report will state the power limitation based on the observed group sizes.
Small cells, inferred gender, proxy outcomes, clustering, and the observational
design will be discussed as limitations. Findings will always be described as
associations and never as proof of bias or proof that bias is absent.
