# Model Card

## Model

The argument classifier uses classical supervised features: TF-IDF word 1–2 grams, optional
character 3–5 grams, and hand features such as token count, numeric count, modality, negation,
hedges, pronouns, and simple POS proportions. Available estimators are balanced Logistic
Regression and LinearSVC.

## Intended use

The model is intended to label sentences in earnings-call questions and answers for aggregate,
interpretable analysis. It supports exploration and feature construction, not automated decisions
about people or companies.

## Training and evaluation

Silver labels come from transparent rules. Gold labels come from manual annotation. The configured
training regimes are gold cross-validation and silver training with gold evaluation. The current
`results/tables/argmining_eval.csv` and gold-set metric outputs are not present in the generated
results directory: **TODO: run and record macro-F1, per-class precision/recall/F1, and confusion
matrix metrics**.

## Failure cases

Expected failures include compound sentences with multiple functions, finance-specific language
outside the lexicons, small gold classes, unusual transcript formatting, and sentences where a
number is present but is not evidence. Rule priority can force one primary label even when flags
for several functions fire.

## Prohibited uses

Do not use this model as a measure of truth, intelligence, credibility, hiring suitability,
individual analyst quality, or evidence of discriminatory intent. It is not a protected-attribute
classifier and should not be used to make individual-level decisions.
